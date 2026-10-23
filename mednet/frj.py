"""FRJ initialization and resource-based route selection for MED-Net."""
import math

def select_codes(occupancy, priority, eligible):
    eligible, priority = tuple(eligible), tuple(priority)
    if len(set(eligible)) != len(eligible) or len(priority) != len(eligible) or set(priority) != set(eligible):
        raise ValueError('priority must be a permutation of eligible positions')
    if isinstance(occupancy, bool) or not math.isfinite(occupancy) or not 0 <= occupancy <= 1:
        raise ValueError('occupancy must be a finite ratio in [0,1]')
    n = 0 if occupancy < .70 else (math.ceil(len(priority)/2) if occupancy < .85 else len(priority))
    deep = set(priority[:n])
    return {position: int(position in deep) for position in eligible}

def sample_gpu_occupancy(device=0):
    raise NotImplementedError('sample_gpu_occupancy is scheduled for a later release; see RELEASE_SCOPE.md.')

def check_requirements(profile, min_accuracy, max_latency_ms, available_memory_bytes):
    """Use a profile measured for the exact route, checkpoint and target device.

    No measured profiles are bundled; this function does not estimate accuracy.
    Accuracy and min_accuracy must use the same metric and reporting scale.
    """
    keys = ('accuracy','latency_ms','peak_memory_bytes')
    if any(k not in profile or not math.isfinite(profile[k]) or profile[k] < 0 for k in keys):
        raise ValueError('Valid measured accuracy/latency/memory profile required')
    if any(not math.isfinite(x) or x < 0 for x in (min_accuracy,max_latency_ms,available_memory_bytes)):
        raise ValueError('Requirements must be finite and nonnegative')
    return profile['accuracy'] >= min_accuracy and profile['latency_ms'] <= max_latency_ms and profile['peak_memory_bytes'] <= available_memory_bytes
