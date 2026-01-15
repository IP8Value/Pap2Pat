### Duty Cycle

Stagioni optimizes duty cycles based on power profiles and fidelity requirements. Figure 7 illustrates how duty cycles vary widely due to the interplay between application power and thermal boundaries. For instance, if thermal boundaries are placed above the temperature response, Stagioni operates at a 100% duty cycle in NSP mode. Conversely, if boundaries lie within steeper rise regions, the system spends more time in CAP mode, leading to lower duty cycles.

### System Power Consumption

Stop-capture-go and seasonal migration significantly reduce system power compared to full-far processing. Figure 8 shows that stop-capture-go consumes the least power as it operates entirely on the near-sensor VPU. Seasonal migration uses both near and far VPUs, consuming more power than stop-capture-go but less than full-far. High fidelity demands lower duty cycles, reducing efficiency and increasing power consumption.

### Overhead

Seasonal migration incurs minimal overhead due to pre-copy techniques, with a switching time of 100 µs, negligible compared to frame capture times. Stop-capture-go's performance impact depends on the duty cycle; higher duty cycles result in minimal performance loss, while lower duty cycles can lead to substantial reductions in effective frame rates. For example, a 30 fps system at 40% duty cycle results in an effective 12 fps.

### Situational Awareness

Stagioni adapts thermal boundaries to ambient conditions and lighting scenarios. Higher ambient temperatures decrease duty cycles by pushing steady-state temperatures further from thermal boundaries, leading to steeper warming phases and gradual cooling phases. Lighting changes fidelity requirements, adjusting \(T_{high}\) and \(T_{low}\). Figure 9b demonstrates smooth temperature variations with light intensity, showcasing Stagioni's dynamic adaptability.

### Conclusions

Stagioni addresses the challenge of thermal noise in near-sensor processing by managing sensor temperature efficiently while optimizing system power. The policies deliver significant power savings (22-53%) with minimal performance loss and adapt smoothly to ambient conditions. This runtime solution marks early steps towards imaging-aware dynamic thermal management, paving the way for more efficient and high-fidelity near-sensor processing systems.