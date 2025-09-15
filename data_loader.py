import numpy as np
import matplotlib.pyplot as plt
import struct
import os
from typing import Tuple, List, Optional, Union


def loadrpc(filename: str, filetype: str = '') -> dict:
    """
    LOADRPC - read an RPC-PRO time history

    This function reads both fixed point and floating point time histories.
    This is a Python translation of the MATLAB loadrpc.m function.

    Parameters:
    -----------
    filename : str
        Name of the RPC-III file to be read (required).
    filetype : str, optional
        Type of file for dialog box display.

    Returns:
    --------
    dict : Dictionary containing all RPC file data with keys:
        'x' : np.ndarray
            Column vector of time history data. If the time history has 5 channels
            and 33792 points per channel, x will have shape (33792, 5).
        'delta_t' : float
            Time increment between sample points.
        'desc' : List[str]
            List containing descriptions for each channel.
        'units' : List[str]
            List containing units for each channel.
        'full_scales' : np.ndarray
            Array of full scale values for each channel.
        'control_mode' : List[str]
            List containing control modes for each channel.
        'map' : List[str]
            List containing logical to physical mapping.
        'npts_frame' : int
            Number of points per frame.
        'npts_grp' : int
            Number of points per group.
        'half' : int
            1 when half frames are present, 0 otherwise.
        'timetype' : str
            RPC3 time type.
        'repeats' : int
            Repeats for each frame.
        'ul' : np.ndarray
            Upper limit for each channel relative to full scale.
        'll' : np.ndarray
            Lower limit for each channel relative to full scale.
        'filename' : str
            Full path and name for the time history.
    """

    # Initialize return dictionary
    result = {
        'x': np.array([]),
        'delta_t': None,
        'desc': [],
        'units': [],
        'full_scales': np.array([]),
        'control_mode': [],
        'map': [],
        'npts_frame': None,
        'npts_grp': None,
        'half': None,
        'timetype': '',
        'repeats': None,
        'ul': np.array([]),
        'll': np.array([]),
        'filename': ''
    }

    # Handle filename input
    if not filename:
        raise ValueError("Filename must be provided")

    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    result['filename'] = filename

    # Read initial header to verify RPC file format
    try:
        with open(filename, 'rb') as fid:
            th = fid.read(512)
    except (IOError, OSError):
        return result

    # Verify that file is an RPC file
    hdr = th[:128]
    key = b'FORMAT'
    if not hdr.startswith(key):
        return result

    # The first 3 records of an RPC-III file are always in the same spot
    # Record size is 128 bytes: 32 for key, 96 for value
    id_key = slice(0, 32)
    id_val = slice(32, 128)

    # Extract number of blocks and parameters
    nblocks = int(th[128 + 32:128 + 128].decode('ascii', errors='ignore').strip().rstrip('\x00'))
    nparms = int(th[2 * 128 + 32:2 * 128 + 128].decode('ascii', errors='ignore').strip().rstrip('\x00'))

    # Read the full header
    with open(filename, 'rb') as fid:
        th = fid.read(512 * nblocks)

    # Initialize storage for parameters
    DESC = {}
    UNITS = {}
    MAP = {}
    CONTROL_MODE = {}
    SCALE = {}
    UPPER_LIMIT = {}
    LOWER_LIMIT = {}

    # General parameters dictionary
    params = {}

    is_time_history = False

    # Parse all parameters
    for i_par in range(nparms):
        start_idx = i_par * 128
        key_bytes = th[start_idx:start_idx + 32]
        val_bytes = th[start_idx + 32:start_idx + 128]

        # Clean up key and value strings
        key = key_bytes.decode('ascii', errors='ignore').strip().rstrip('\x00').upper()
        value = val_bytes.decode('ascii', errors='ignore').strip().rstrip('\x00')

        if not key:
            continue

        # Skip problematic keys for now
        if any(char in key for char in ['/', '(', ',']):
            continue

        # Handle channel-specific parameters (those with dots)
        if '.' in key:
            try:
                # Find the last underscore and get the channel number
                chan_num_str = key.rsplit('_', 1)[-1]
                chan_num = int(chan_num_str)

                if key.startswith('DESC'):
                    DESC[chan_num] = value
                elif key.startswith('UNIT'):
                    UNITS[chan_num] = value
                elif key.startswith('MAP'):
                    MAP[chan_num] = value
                elif key.startswith('CON'):
                    CONTROL_MODE[chan_num] = value
                elif key.startswith('SCAL'):
                    SCALE[chan_num] = float(value)
                elif key.startswith('UPPE'):
                    UPPER_LIMIT[chan_num] = float(value)
                elif key.startswith('LOWE'):
                    LOWER_LIMIT[chan_num] = float(value)
            except (ValueError, IndexError):
                # Ignore keys that don't parse correctly
                pass
        else:
            # General parameters
            params[key] = value

            # Check if this is a time history file
            if key == 'FILE_TYPE' and value == 'TIME_HISTORY':
                is_time_history = True

    # Return if file is not a time history file
    if not is_time_history:
        return result

    # Extract required parameters
    try:
        nchan = int(params['CHANNELS'])
        npts_frame = int(params['PTS_PER_FRAME'])
        npts_grp = int(params['PTS_PER_GROUP'])
        delta_t = float(params['DELTA_T'])
        nframes = int(params['FRAMES'])
        timetype = params.get('TIME_TYPE', '')
        repeats = int(params.get('REPEATS', '0'))

        # Handle half frames
        half_frames_str = params.get('HALF_FRAMES', 'NO').upper().strip()
        half = 1 if half_frames_str == 'YES' else 0
        nframes += half

        # Extract data type
        data_type = params.get('DATA_TYPE', 'FIXED_POINT')

    except (KeyError, ValueError):
        return result

    # Build per-channel info
    ichan = list(range(1, nchan + 1))

    scales = np.array([SCALE.get(i, 1.0) for i in ichan])
    desc = [DESC.get(i, '') for i in ichan]
    units = [UNITS.get(i, '') for i in ichan]
    map_data = [MAP.get(i, '') for i in ichan]
    ul = np.array([UPPER_LIMIT.get(i, 0.0) for i in ichan])
    ll = np.array([LOWER_LIMIT.get(i, 0.0) for i in ichan])
    control_mode = [CONTROL_MODE.get(i, '') for i in ichan]

    full_scales = 32752.0 * scales

    # Read the actual time history data
    npts_total = nframes * npts_frame
    x = np.zeros((npts_total, nchan), dtype=np.float64)

    npts_read_per_group = nchan * npts_grp

    with open(filename, 'rb') as fid:
        # Skip header
        fid.seek(512 * nblocks)

        # Determine read format
        if data_type == 'FLOATING_POINT':
            read_format = 'f'  # float32
            bytes_per_sample = 4
            do_fp = True
        else:
            read_format = 'h'  # int16
            bytes_per_sample = 2
            do_fp = False

        ind_start = 0
        n_full_groups = npts_total // npts_grp

        for _ in range(n_full_groups):
            data_bytes = fid.read(npts_read_per_group * bytes_per_sample)
            if len(data_bytes) < npts_read_per_group * bytes_per_sample:
                break

            # Unpack raw data
            xr_flat = np.array(struct.unpack(f'<{npts_read_per_group}{read_format}', data_bytes))


            # Reshape using 'F' (Fortran/column-major) order to match MATLAB's memory layout.
            # The data is stored as a block for channel 1, then a block for channel 2, etc.
            xr = xr_flat.reshape((npts_grp, nchan), order='F')

            if do_fp:
                x[ind_start:ind_start + npts_grp, :] = xr
            else:
                # Apply scaling for fixed point data
                x[ind_start:ind_start + npts_grp, :] = xr * scales

            ind_start += npts_grp

        # Handle remaining points if any
        npts_remaining = npts_total % npts_grp
        if npts_remaining > 0:
            # We need to read a full group, even if we only use part of it
            data_bytes = fid.read(npts_read_per_group * bytes_per_sample)

            if len(data_bytes) >= npts_read_per_group * bytes_per_sample:
                xr_flat = np.array(struct.unpack(f'<{npts_read_per_group}{read_format}', data_bytes))

                # Also apply Fortran ordering here
                xr = xr_flat.reshape((npts_grp, nchan), order='F')

                if do_fp:
                    x[ind_start:ind_start + npts_remaining, :] = xr[:npts_remaining, :]
                else:
                    x[ind_start:ind_start + npts_remaining, :] = xr[:npts_remaining, :] * scales

    # Populate result dictionary
    result.update({
        'x': x,
        'delta_t': delta_t,
        'desc': desc,
        'units': units,
        'full_scales': full_scales,
        'control_mode': control_mode,
        'map': map_data,
        'npts_frame': npts_frame,
        'npts_grp': npts_grp,
        'half': half,
        'timetype': timetype,
        'repeats': repeats,
        'ul': ul,
        'll': ll
    })

    return result
