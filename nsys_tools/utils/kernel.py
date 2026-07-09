"""
Kernel-side SQL.
Author: yezhengmaolove@gmail.com
"""

# =============================================================================
# SQL
# =============================================================================

KERNEL_SQL = """
SELECT
    k.start         AS gpu_start,
    k.end           AS gpu_end,
    k.end - k.start AS gpu_dur_ns,
    s_k.value       AS kernel_name,
    s_api.value     AS launch_api,
    r.start         AS api_start,
    r.end           AS api_end,
    k.deviceId,
    k.streamId,
    k.gridX, k.gridY, k.gridZ,
    k.blockX, k.blockY, k.blockZ,
    k.registersPerThread,
    k.staticSharedMemory,
    k.dynamicSharedMemory
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId
JOIN StringIds s_k                 ON s_k.id = k.shortName
JOIN StringIds s_api               ON s_api.id = r.nameId
"""
