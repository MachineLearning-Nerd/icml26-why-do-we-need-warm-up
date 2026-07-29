# Limitations and deviations

- The route uses three 30-step early trajectories and batch 8, not full
  ImageNet32 training.
- It samples the first official training member rather than all ten members.
- It uses float32 CPU. These deviations reduce power, so a nonsignificant
  result is `BLOCKED`, never presented as falsification.
