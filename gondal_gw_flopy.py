"""
gondal_gw_flopy.py
==================
FloPy (MODFLOW-6) replication of the Gondal Groundwater Model.

Original model created with ModelMuse 5.4.0.0 on 19-03-2026.

This script reads the large grid arrays (vertices, cell topology, initial heads,
recharge rates, constant-head boundaries, ghost-node corrections) directly from
the original MODFLOW-6 input files and builds an equivalent FloPy model that you
can modify and re-run without touching those files.

HOW TO USE
----------
1. Edit the "USER-MODIFIABLE PARAMETERS" section (Section 2) to change any
   hydraulic conductivity, storage, boundary-condition, or time-stepping value.
2. Run the script:
       python gondal_gw_flopy.py
   FloPy writes the new MODFLOW-6 input files to the folder set in ``model_ws``.
3. Run MODFLOW-6 on the generated files (requires mf6 executable on PATH or set
   ``exe_name`` to the full path).

REQUIREMENTS
------------
  pip install flopy numpy
"""

import os
import re
import numpy as np

try:
    import flopy
    import flopy.mf6 as mf6
except ImportError:
    raise ImportError("FloPy is required.  Install with:  pip install flopy")

# ===========================================================================
# SECTION 1 – PATHS
# ===========================================================================

# Directory that contains the original MODFLOW-6 input files
INPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Output directory where FloPy will write the new MODFLOW-6 input files
model_ws = os.path.join(INPUT_DIR, "flopy_output")

# MODFLOW-6 executable name (must be on PATH or provide full path)
exe_name = "mf6"

# ===========================================================================
# SECTION 2 – USER-MODIFIABLE PARAMETERS
# ===========================================================================
# Edit the values in this section to change the model without touching any
# other part of the script.

# ---------------------------------------------------------------------------
# 2a.  Temporal Discretization (TDIS)
# ---------------------------------------------------------------------------
# time_units: 'days', 'seconds', 'minutes', 'hours', or 'years'
time_units = "days"

# perioddata: list of (perlen, nstp, tsmult) tuples – one per stress period
# Period 1 is set to STEADY-STATE; periods 2-8 are TRANSIENT (see STO block).
perioddata = [
    (1.0,  1,  1.0),   # SP 1  – 1 day,  1 step  (steady-state initialisation)
    (5.0,  8,  1.1),   # SP 2  – 5 days, 8 steps
    (1.0, 50,  1.0),   # SP 3  – 1 day, 50 steps
    (1.0, 50,  1.0),   # SP 4  – 1 day, 50 steps
    (1.0, 50,  1.0),   # SP 5  – 1 day, 50 steps
    (1.0, 50,  1.0),   # SP 6  – 1 day, 50 steps
    (9.0, 11,  1.1),   # SP 7  – 9 days, 11 steps
    (4.0,  7,  1.1),   # SP 8  – 4 days,  7 steps
]

# ---------------------------------------------------------------------------
# 2b.  Node-Property Flow (NPF) – hydraulic conductivity
# ---------------------------------------------------------------------------
# One value per layer (12 layers total).
# Layer 1    = uppermost / unconfined aquifer
# Layers 2–11 = confined aquifer
# Layer 12   = aquitard / basement

# Horizontal hydraulic conductivity  [m/d]
k_hk = [
    1.0,      # Layer  1
    0.8,      # Layer  2
    0.8,      # Layer  3
    0.8,      # Layer  4
    0.8,      # Layer  5
    0.8,      # Layer  6
    0.8,      # Layer  7
    0.8,      # Layer  8
    0.8,      # Layer  9
    0.8,      # Layer 10
    0.8,      # Layer 11
    1.0e-6,   # Layer 12 (aquitard)
]

# Vertical hydraulic conductivity  [m/d]  (K33 in MODFLOW-6 notation)
k_vk = [
    0.1,      # Layer  1
    0.08,     # Layer  2
    0.08,     # Layer  3
    0.08,     # Layer  4
    0.08,     # Layer  5
    0.08,     # Layer  6
    0.08,     # Layer  7
    0.08,     # Layer  8
    0.08,     # Layer  9
    0.08,     # Layer 10
    0.08,     # Layer 11
    1.0e-7,   # Layer 12 (aquitard)
]

# Horizontal anisotropy – K22 (y-direction K relative to x-direction K)  [m/d]
k22 = [
    1.0,      # Layer  1
    0.8,      # Layer  2
    0.8,      # Layer  3
    0.8,      # Layer  4
    0.8,      # Layer  5
    0.8,      # Layer  6
    0.8,      # Layer  7
    0.8,      # Layer  8
    0.8,      # Layer  9
    0.8,      # Layer 10
    0.8,      # Layer 11
    1.0e-6,   # Layer 12 (aquitard)
]

# Cell type: 1 = convertible (unconfined), 0 = confined
# (matches ICELLTYPE in NPF and ICONVERT in STO)
icelltype = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Layer 1 unconfined

# ---------------------------------------------------------------------------
# 2c.  Storage (STO)
# ---------------------------------------------------------------------------
# Specific storage  [1/m]  – one value per layer
ss = [1.0e-4] * 12

# Specific yield  [-]  – one value per layer (used only for convertible layers)
sy = [0.02] * 12

# ---------------------------------------------------------------------------
# 2d.  Recharge (RCH) – uniform rate applied to all 21,513 Layer-1 cells
# ---------------------------------------------------------------------------
# Units: m/d.  Derived from the original RCH package.
# SP 1 and SP 2 have zero recharge (dry / pre-monsoon);
# SP 8 returns to zero (post-monsoon recession).
rch_rates = {
    1: 0.0,             # SP 1 – steady state
    2: 0.0,             # SP 2
    3: 1.990960e-2,     # SP 3
    4: 2.140195e-2,     # SP 4
    5: 2.289366e-2,     # SP 5
    6: 9.222197e-3,     # SP 6
    7: 2.949371e-4,     # SP 7
    8: 0.0,             # SP 8
}

# ---------------------------------------------------------------------------
# 2e.  IMS Iterative Solver settings
# ---------------------------------------------------------------------------
outer_dvclose = 1.0e-2     # Nonlinear convergence criterion  [m]
outer_maximum = 500        # Max outer (Newton) iterations
inner_maximum = 350        # Max inner (linear solver) iterations
inner_dvclose = 1.0e-3     # Inner convergence criterion (head change)  [m]
inner_rclose  = 1.0e-1     # Inner convergence criterion (residual)

# ---------------------------------------------------------------------------
# 2f.  River time-series file name
# ---------------------------------------------------------------------------
# The river stages are stored in the original time-series file.
# The script copies a reference to that file into the new package.
ts_filename = "TS_19_chd.riv.Group1.ts"

# ===========================================================================
# SECTION 3 – FILE PARSERS
# (read large arrays from the original input files – do not edit unless the
#  file format changes)
# ===========================================================================

def _read_layer_array(fobj, ncpl):
    """
    Read a single-layer array from *fobj*.

    This function reads the descriptor line (INTERNAL … or CONSTANT …)
    from the current position in *fobj* and then reads the data values.

    Parameters
    ----------
    fobj  : open file object positioned at the descriptor line
    ncpl  : number of values expected for INTERNAL arrays

    Returns (values, next_line) where *next_line* is the first line that
    was read but NOT consumed (i.e. belongs to the next token).
    """
    # Read the descriptor line (INTERNAL … or CONSTANT …)
    while True:
        desc = fobj.readline()
        if not desc:
            return np.array([]), ""
        stripped = desc.strip()
        if stripped and not stripped.startswith("#"):
            break

    upper = stripped.upper().split("#")[0].strip()

    if upper.startswith("CONSTANT"):
        val = float(upper.split()[1])
        return np.full(ncpl, val), ""
    elif upper.startswith("INTERNAL"):
        vals = []
        while len(vals) < ncpl:
            line = fobj.readline()
            if not line:
                break
            clean = line.split("#")[0]
            tokens = clean.split()
            # Stop if we hit a new keyword (non-numeric first token)
            if tokens and not _is_float(tokens[0]):
                return np.array(vals[:ncpl]), line
            vals.extend(float(v) for v in tokens)
        return np.array(vals[:ncpl]), ""
    else:
        # Unknown descriptor — return the line for the caller to process
        return np.array([]), desc


def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_disv(filepath, nlay, ncpl, nvert):
    """
    Parse a MODFLOW-6 DISV input file.

    Returns
    -------
    top      : np.ndarray  shape (ncpl,)
    botm     : np.ndarray  shape (nlay, ncpl)
    vertices : list of [iv, xv, yv]  (1-based vertex indices)
    cell2d   : list of [icell2d, xc, yc, ncvert, v1, v2, …]  (1-based)
    """
    print(f"  Parsing DISV file: {filepath}")
    top      = None
    botm     = []
    vertices = []
    cell2d   = []

    with open(filepath, "r") as f:
        section = None
        for line in f:
            stripped = line.strip()
            upper = stripped.upper()

            # Section markers
            if upper.startswith("BEGIN GRIDDATA"):
                section = "GRIDDATA"
                continue
            elif upper.startswith("END GRIDDATA"):
                section = None
                continue
            elif upper.startswith("BEGIN VERTICES"):
                section = "VERTICES"
                continue
            elif upper.startswith("END VERTICES"):
                section = None
                continue
            elif upper.startswith("BEGIN CELL2D"):
                section = "CELL2D"
                continue
            elif upper.startswith("END CELL2D"):
                section = None
                continue
            elif upper.startswith("BEGIN") or upper.startswith("END"):
                section = None
                continue

            if not stripped or stripped.startswith("#"):
                continue

            clean_upper = upper.split("#")[0].strip()

            if section == "GRIDDATA":
                if clean_upper.startswith("TOP"):
                    arr, _ = _read_layer_array(f, ncpl)
                    top = arr
                elif clean_upper.startswith("BOTM"):
                    # "BOTM LAYERED" – read nlay layers sequentially
                    for _lay in range(nlay):
                        arr, leftover = _read_layer_array(f, ncpl)
                        botm.append(arr)
                # IDOMAIN / angle1/2/3 are CONSTANT – skip automatically

            elif section == "VERTICES":
                parts = stripped.split("#")[0].split()
                if len(parts) >= 3:
                    try:
                        iv = int(parts[0])
                        xv = float(parts[1])
                        yv = float(parts[2])
                        vertices.append([iv, xv, yv])
                    except ValueError:
                        pass

            elif section == "CELL2D":
                parts = stripped.split("#")[0].split()
                if len(parts) >= 5:
                    try:
                        icell  = int(parts[0])
                        xc     = float(parts[1])
                        yc     = float(parts[2])
                        ncvert = int(parts[3])
                        verts  = [int(v) for v in parts[4:4 + ncvert]]
                        cell2d.append([icell, xc, yc, ncvert] + verts)
                    except (ValueError, IndexError):
                        pass

    botm_arr = np.array(botm)  # shape (nlay, ncpl)
    print(f"    TOP shape  : {top.shape}")
    print(f"    BOTM shape : {botm_arr.shape}")
    print(f"    Vertices   : {len(vertices)}")
    print(f"    Cell2D     : {len(cell2d)}")
    return top, botm_arr, vertices, cell2d


def parse_ic(filepath, nlay, ncpl):
    """
    Parse a MODFLOW-6 IC (Initial Conditions) file.

    Returns
    -------
    strt : np.ndarray  shape (nlay, ncpl)  – starting heads [m]
    """
    print(f"  Parsing IC file: {filepath}")
    strt = []

    with open(filepath, "r") as f:
        section = None
        for line in f:
            stripped = line.strip()
            upper = stripped.upper()

            if upper.startswith("BEGIN GRIDDATA"):
                section = "GRIDDATA"
                continue
            elif upper.startswith("END GRIDDATA"):
                section = None
                continue
            elif upper.startswith("BEGIN") or upper.startswith("END"):
                section = None
                continue

            if not stripped or stripped.startswith("#"):
                continue

            if section == "GRIDDATA":
                clean_upper = upper.split("#")[0].strip()
                if clean_upper.startswith("STRT"):
                    # STRT LAYERED
                    for _lay in range(nlay):
                        arr, _ = _read_layer_array(f, ncpl)
                        strt.append(arr)

    strt_arr = np.array(strt)
    print(f"    STRT shape : {strt_arr.shape}")
    return strt_arr


def parse_chd(filepath):
    """
    Parse a MODFLOW-6 CHD (Constant-Head) file.

    Returns
    -------
    chd_spd : dict  { period_index(0-based) : list of [layer-1, cell2d-1, head, iface, bname] }
    """
    print(f"  Parsing CHD file: {filepath}")
    chd_spd = {}
    current_period = None

    with open(filepath, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            upper = stripped.upper()
            if upper.startswith("BEGIN PERIOD"):
                current_period = int(stripped.split()[2]) - 1  # 0-based
                chd_spd[current_period] = []
            elif upper.startswith("END PERIOD"):
                current_period = None
            elif current_period is not None:
                data = stripped.split("#")[0].strip()
                if not data:
                    continue
                parts = data.split()
                if len(parts) >= 3:
                    lay   = int(parts[0]) - 1   # 0-based
                    node  = int(parts[1]) - 1   # 0-based
                    head  = float(parts[2])
                    iface = int(parts[3]) if len(parts) > 3 else 0
                    bname = parts[4].strip("'\"") if len(parts) > 4 else ""
                    chd_spd[current_period].append(
                        [(lay, node), head, iface, bname]
                    )

    for p, lst in chd_spd.items():
        print(f"    CHD period {p+1}: {len(lst)} entries")
    return chd_spd


def parse_riv(filepath):
    """
    Parse a MODFLOW-6 RIV (River) file.

    Returns
    -------
    riv_spd : dict  { period_index(0-based) :
                      list of [(lay-1, node-1), stage_ts_name, cond, rbot, iface, bname] }
    ts_file : str   name of the linked time-series file (or None)
    """
    print(f"  Parsing RIV file: {filepath}")
    riv_spd = {}
    ts_file = None
    current_period = None

    with open(filepath, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            upper = stripped.upper()
            if "TS6 FILEIN" in upper:
                ts_file = stripped.split()[-1].strip("'\"")
            elif upper.startswith("BEGIN PERIOD"):
                current_period = int(stripped.split()[2]) - 1
                riv_spd[current_period] = []
            elif upper.startswith("END PERIOD"):
                current_period = None
            elif current_period is not None:
                data = stripped.split("#")[0].strip()
                if not data:
                    continue
                parts = data.split()
                if len(parts) >= 5:
                    lay   = int(parts[0]) - 1
                    node  = int(parts[1]) - 1
                    stage = parts[2]              # may be a TS name like 'TS_1'
                    cond  = float(parts[3])
                    rbot  = float(parts[4])
                    iface = int(parts[5]) if len(parts) > 5 else 0
                    bname = parts[6].strip("'\"") if len(parts) > 6 else ""
                    riv_spd[current_period].append(
                        [(lay, node), stage, cond, rbot, iface, bname]
                    )

    for p, lst in riv_spd.items():
        print(f"    RIV period {p+1}: {len(lst)} entries")
    return riv_spd, ts_file


def parse_rch(filepath, ncpl):
    """
    Parse a MODFLOW-6 RCH (Recharge) file.

    Because each stress period contains one entry per cell (21,513 cells)
    the function extracts per-cell values; if a period is uniform it stores
    a scalar, otherwise a full array is stored.

    Returns
    -------
    rch_spd : dict  { period_index(0-based) :
                      list of [(lay-1, node-1), rate, iface, bname] }
    """
    print(f"  Parsing RCH file: {filepath}")
    rch_spd = {}
    current_period = None

    with open(filepath, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            upper = stripped.upper()
            if upper.startswith("BEGIN PERIOD"):
                current_period = int(stripped.split()[2]) - 1
                rch_spd[current_period] = []
            elif upper.startswith("END PERIOD"):
                current_period = None
            elif current_period is not None:
                data = stripped.split("#")[0].strip()
                if not data:
                    continue
                parts = data.split()
                if len(parts) >= 3 and parts[0].isdigit():
                    try:
                        lay   = int(parts[0]) - 1
                        node  = int(parts[1]) - 1
                        rate  = float(parts[2])
                        iface = int(parts[3]) if len(parts) > 3 else 0
                        bname = parts[4].strip("'\"") if len(parts) > 4 else ""
                        rch_spd[current_period].append(
                            [(lay, node), rate, iface, bname]
                        )
                    except (ValueError, IndexError):
                        pass

    for p, lst in rch_spd.items():
        print(f"    RCH period {p+1}: {len(lst)} entries")
    return rch_spd


def parse_gnc(filepath):
    """
    Parse a MODFLOW-6 GNC (Ghost-Node Correction) file.

    The original file uses 1-based (layer, node) indices.
    This function converts them to FloPy's 0-based convention and
    replaces padding entries (0, 0) with (-1, -1) as required by FloPy.

    Returns
    -------
    numgnc    : int
    numalphaj : int
    gncdata   : list of tuples  [((lay_i, node_i), (lay_j, node_j),
                                   (ghost_1), …, alpha1, …)]
                compatible with flopy.mf6.ModflowGwfgnc
    """
    print(f"  Parsing GNC file: {filepath}")
    numgnc    = 0
    numalphaj = 0
    gncdata   = []
    in_gnc    = False

    with open(filepath, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            upper = stripped.upper()
            if upper.startswith("NUMGNC"):
                numgnc = int(stripped.split()[1])
            elif upper.startswith("NUMALPHAJ"):
                numalphaj = int(stripped.split()[1])
            elif upper.startswith("BEGIN GNCDATA"):
                in_gnc = True
            elif upper.startswith("END GNCDATA"):
                in_gnc = False
            elif in_gnc:
                data = stripped.split("#")[0].strip()
                if not data:
                    continue
                parts = data.split()
                # GNC row: (lay_i, node_i), (lay_j, node_j),
                #          numalphaj pairs of (ghost_lay, ghost_node),
                #          then numalphaj alpha values
                # Total integers = 2*(2 + numalphaj); total floats = numalphaj
                n_int_pairs = 2 + numalphaj   # i-cell + j-cell + ghost cells
                n_int_vals  = 2 * n_int_pairs
                n_float_vals = numalphaj

                if len(parts) < n_int_vals + n_float_vals:
                    continue

                # Convert 1-based (lay, node) to 0-based, (0,0) → (-1,-1)
                cells = []
                for k in range(n_int_pairs):
                    lay  = int(parts[2 * k])
                    node = int(parts[2 * k + 1])
                    if lay == 0 and node == 0:
                        cells.append((-1, -1))
                    else:
                        cells.append((lay - 1, node - 1))

                alphas = [float(parts[n_int_vals + a])
                          for a in range(n_float_vals)]

                # Row: cellidn, cellidm, *ghost_cells, *alphas
                row = cells + alphas
                gncdata.append(row)

    print(f"    GNC entries: {len(gncdata)}  (NUMALPHAJ={numalphaj})")
    return numgnc, numalphaj, gncdata


def parse_ts_file(filepath):
    """
    Parse a MODFLOW-6 time-series (TS6) file.

    Returns a dict with keys:
        'names'   : list of TS names
        'methods' : list of interpolation methods
        'sfacs'   : list of scale factors
        'data'    : list of (time, val1, val2, …) tuples
    """
    print(f"  Parsing TS file: {filepath}")
    names   = []
    methods = []
    sfacs   = []
    ts_data = []
    in_attr = False
    in_ts   = False

    with open(filepath, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            upper = stripped.upper()
            if upper.startswith("BEGIN ATTRIBUTES"):
                in_attr = True
            elif upper.startswith("END ATTRIBUTES"):
                in_attr = False
            elif upper.startswith("BEGIN TIMESERIES"):
                in_ts = True
            elif upper.startswith("END TIMESERIES"):
                in_ts = False
            elif in_attr:
                if upper.startswith("NAMES"):
                    names = stripped.split()[1:]
                elif upper.startswith("METHODS"):
                    methods = stripped.split()[1:]
                elif upper.startswith("SFACS"):
                    sfacs = [float(v) for v in stripped.split()[1:]]
            elif in_ts:
                vals = [float(v) for v in stripped.split()]
                if vals:
                    ts_data.append(vals)

    print(f"    TS names: {len(names)}, time records: {len(ts_data)}")
    return {"names": names, "methods": methods, "sfacs": sfacs, "data": ts_data}


# ===========================================================================
# SECTION 4 – BUILD THE FLOPY MODEL
# ===========================================================================

def build_model():
    os.makedirs(model_ws, exist_ok=True)

    # -------------------------------------------------------------------
    # 4a. Grid dimensions (from DISV file)
    # -------------------------------------------------------------------
    nlay  = 12
    ncpl  = 21513
    nvert = 21942

    # -------------------------------------------------------------------
    # 4b. Parse large arrays from original files
    # -------------------------------------------------------------------
    print("\n=== Parsing original MODFLOW-6 input files ===")

    disv_file = os.path.join(INPUT_DIR, "TS_19_chd.disv")
    ic_file   = os.path.join(INPUT_DIR, "TS_19_chd.ic")
    chd_file  = os.path.join(INPUT_DIR, "TS_19_chd.chd")
    riv_file  = os.path.join(INPUT_DIR, "TS_19_chd.riv")
    rch_file  = os.path.join(INPUT_DIR, "TS_19_chd.rch")
    gnc_file  = os.path.join(INPUT_DIR, "TS_19_chd.gnc")
    ts_file   = os.path.join(INPUT_DIR, ts_filename)

    top, botm, vertices, cell2d = parse_disv(disv_file, nlay, ncpl, nvert)
    strt                        = parse_ic(ic_file, nlay, ncpl)
    chd_spd                     = parse_chd(chd_file)
    riv_spd, _ts_ref            = parse_riv(riv_file)
    rch_spd                     = parse_rch(rch_file, ncpl)
    numgnc, numalphaj, gncdata  = parse_gnc(gnc_file)
    ts_info                     = parse_ts_file(ts_file)

    print("\n=== Building FloPy MODFLOW-6 model ===")

    # -------------------------------------------------------------------
    # 4c. Simulation
    # -------------------------------------------------------------------
    sim = mf6.MFSimulation(
        sim_name="TS_19_chd",
        version="mf6",
        exe_name=exe_name,
        sim_ws=model_ws,
        memory_print_option="summary",
        print_input=True,
    )

    # -------------------------------------------------------------------
    # 4d. Temporal Discretization (TDIS)
    # -------------------------------------------------------------------
    tdis = mf6.ModflowTdis(
        sim,
        time_units=time_units,
        nper=len(perioddata),
        perioddata=perioddata,
    )
    print("  TDIS created")

    # -------------------------------------------------------------------
    # 4e. Iterative Model Solution (IMS)
    # -------------------------------------------------------------------
    ims = mf6.ModflowIms(
        sim,
        print_option="summary",
        complexity="complex",
        no_ptcrecord=["FIRST"],
        ats_outer_maximum_fraction=1.0 / 3.0,
        outer_dvclose=outer_dvclose,
        outer_maximum=outer_maximum,
        under_relaxation="NONE",
        backtracking_number=20,
        inner_maximum=inner_maximum,
        inner_dvclose=inner_dvclose,
        rcloserecord=inner_rclose,
        linear_acceleration="BICGSTAB",
        csv_outer_output_filerecord="TS_19_chd.OuterSolution.CSV",
        csv_inner_output_filerecord="TS_19_chd.InnerSolution.CSV",
    )
    print("  IMS created")

    # -------------------------------------------------------------------
    # 4f. Groundwater-Flow Model (GWF)
    # -------------------------------------------------------------------
    gwf = mf6.ModflowGwf(
        sim,
        modelname="MODFLOW",
        model_nam_file="TS_19_chd.nam",
        newtonoptions="NEWTON",
        print_input=True,
        save_flows=True,
    )
    sim.register_ims_package(ims, [gwf.name])
    print("  GWF model created")

    # -------------------------------------------------------------------
    # 4g. Discretization by Vertices (DISV)
    # -------------------------------------------------------------------
    disv_pkg = mf6.ModflowGwfdisv(
        gwf,
        length_units="meters",
        nlay=nlay,
        ncpl=ncpl,
        nvert=nvert,
        top=top,
        botm=botm,
        vertices=vertices,
        cell2d=cell2d,
    )
    print("  DISV created")

    # -------------------------------------------------------------------
    # 4h. Initial Conditions (IC)
    # -------------------------------------------------------------------
    ic_pkg = mf6.ModflowGwfic(
        gwf,
        strt=strt,
    )
    print("  IC created")

    # -------------------------------------------------------------------
    # 4i. Node Property Flow (NPF)
    # -------------------------------------------------------------------
    npf_pkg = mf6.ModflowGwfnpf(
        gwf,
        save_flows=True,
        save_specific_discharge=True,
        save_saturation=True,
        icelltype=icelltype,
        k=k_hk,
        k33=k_vk,
        k22=k22,
    )
    print("  NPF created")

    # -------------------------------------------------------------------
    # 4j. Storage (STO)
    # -------------------------------------------------------------------
    # Build stress-period data: {0: "STEADY-STATE", 1: "TRANSIENT", …}
    sto_spd = {0: "STEADY-STATE"}
    for i in range(1, len(perioddata)):
        sto_spd[i] = "TRANSIENT"

    sto_pkg = mf6.ModflowGwfsto(
        gwf,
        save_flows=True,
        iconvert=icelltype,          # same flag as ICELLTYPE
        ss=ss,
        sy=sy,
        steady_state={0: True},
        transient={i: True for i in range(1, len(perioddata))},
    )
    print("  STO created")

    # -------------------------------------------------------------------
    # 4k. Output Control (OC)
    # -------------------------------------------------------------------
    oc_spd = {}
    for i in range(len(perioddata)):
        oc_spd[(i, 0)] = ["SAVE HEAD", "SAVE BUDGET", "PRINT BUDGET"]

    oc_pkg = mf6.ModflowGwfoc(
        gwf,
        budget_filerecord="TS_19_chd.cbc",
        head_filerecord="TS_19_chd.bhd",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("BUDGET", "ALL")],
    )
    print("  OC created")

    # -------------------------------------------------------------------
    # 4l. Constant Head (CHD)
    # -------------------------------------------------------------------
    # Convert parsed CHD data into FloPy stress-period dict format:
    # key = 0-based period; value = list of [(lay,node), head, aux, bname]
    flopy_chd_spd = {}
    for per, rows in chd_spd.items():
        flopy_chd_spd[per] = [
            [r[0], r[1], r[2], r[3]] for r in rows
        ]

    chd_pkg = mf6.ModflowGwfchd(
        gwf,
        auxiliary=["IFACE"],
        boundnames=True,
        print_input=True,
        save_flows=True,
        maxbound=max(len(v) for v in flopy_chd_spd.values()),
        stress_period_data=flopy_chd_spd,
    )
    print(f"  CHD created ({len(chd_spd)} periods)")

    # -------------------------------------------------------------------
    # 4m. River (RIV) with Time Series
    # -------------------------------------------------------------------
    # The river stage values reference time-series names (e.g. 'TS_1').
    # We pass the same TS file used in the original model.

    flopy_riv_spd = {}
    for per, rows in riv_spd.items():
        flopy_riv_spd[per] = [
            [r[0], r[1], r[2], r[3], r[4], r[5]] for r in rows
        ]

    riv_pkg = mf6.ModflowGwfriv(
        gwf,
        auxiliary=["IFACE"],
        boundnames=True,
        print_input=True,
        save_flows=True,
        timeseries={
            "filename": ts_filename,
            "time_series_namerecord": ts_info["names"],
            "interpolation_methodrecord": ts_info["methods"],
            "sfacrecord": ts_info["sfacs"],
            "timeseries": ts_info["data"],
        },
        maxbound=max(len(v) for v in flopy_riv_spd.values()) if flopy_riv_spd else 130,
        stress_period_data=flopy_riv_spd,
    )
    print(f"  RIV created ({len(riv_spd)} periods)")

    # -------------------------------------------------------------------
    # 4n. Recharge (RCH)
    # -------------------------------------------------------------------
    # The original file has per-cell recharge entries; use the parsed data
    # directly so that any future per-cell customisation is preserved.
    flopy_rch_spd = {}
    for per, rows in rch_spd.items():
        flopy_rch_spd[per] = [
            [r[0], r[1], r[2], r[3]] for r in rows
        ]

    rch_pkg = mf6.ModflowGwfrch(
        gwf,
        auxiliary=["IFACE"],
        boundnames=True,
        print_input=True,
        save_flows=True,
        fixed_cell=True,
        maxbound=ncpl,
        stress_period_data=flopy_rch_spd,
    )
    print(f"  RCH created ({len(rch_spd)} periods)")

    # -------------------------------------------------------------------
    # 4o. Ghost-Node Correction (GNC)
    # -------------------------------------------------------------------
    # gncdata rows are already in FloPy 0-based format from parse_gnc().
    gnc_pkg = mf6.ModflowGwfgnc(
        gwf,
        print_input=True,
        numgnc=numgnc,
        numalphaj=numalphaj,
        gncdata=gncdata,
    )
    print(f"  GNC created ({numgnc} entries)")

    # -------------------------------------------------------------------
    # 4p. Write the model files
    # -------------------------------------------------------------------
    print(f"\n=== Writing FloPy model files to: {model_ws} ===")
    sim.write_simulation()
    print("  Done writing MODFLOW-6 input files.")

    return sim, gwf


# ===========================================================================
# SECTION 5 – RUN THE MODEL  (optional)
# ===========================================================================

def run_model(sim):
    """
    Run MODFLOW-6 using the simulation object.

    Requires the ``mf6`` executable to be available on PATH or set ``exe_name``
    to the full path of the executable in Section 1.
    """
    print(f"\n=== Running MODFLOW-6 ({exe_name}) ===")
    success, buff = sim.run_simulation()
    if success:
        print("  MODFLOW-6 run completed successfully.")
    else:
        print("  MODFLOW-6 run FAILED.  Check listing file for details.")
        for line in buff:
            print("   ", line)
    return success


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    sim, gwf = build_model()

    # Uncomment the line below to run MODFLOW-6 immediately after writing files:
    # run_model(sim)

    print("\nAll done.  Model files written to:", model_ws)
    print("To run the model, either:")
    print("  1.  Uncomment 'run_model(sim)' at the bottom of this script, or")
    print("  2.  Run:  cd", model_ws, "&&", exe_name)
