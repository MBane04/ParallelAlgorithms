NSYS_PROFILING_GPU_GL_ENABLE=1 \
NSYS_GPU_PROFILER_SYNC_DELAY_MS=100 \
nsys profile \
  --trace=cuda,nvtx,opengl \
  --stats=true \
  --output=final_report \
  ./demo 500 1
#--trace tells nsys what to profile
#cuda-hw shows hardware counters
#nvtx shows nvtx events
#osrt shows os runtime events