Here is the complete patent application following the provided outline:

## FIELD OF THE INVENTION

The present invention relates generally to assistive technologies for individuals with mobility impairments, and more particularly to a system and method for maintaining trunk stability in manual wheelchair users through real-time detection of destabilizing events and responsive neuromuscular stimulation. The invention combines inertial measurement technology with functional neuromuscular stimulation (FNS) to automatically activate paralyzed trunk and hip muscles when potentially destabilizing situations are detected during wheelchair use. This system addresses the critical need for improved postural stability in individuals with spinal cord injuries (SCI) who rely on manual wheelchairs for mobility, while avoiding the limitations of current passive restraint systems. The invention represents a significant advancement in neuroprosthetic technology by providing active stabilization that responds dynamically to real-world conditions encountered during wheelchair propulsion.

## BACKGROUND

Individuals with spinal cord injuries face substantial challenges in maintaining trunk stability during manual wheelchair use. Current methods for addressing postural instability in wheelchair users primarily involve passive restraint systems such as seat belts, specialized cushions, or fixed supports. These conventional approaches suffer from multiple disadvantages including restricted voluntary movement, pressure ulcers, skin damage, and psychological impacts from perceived loss of independence. While powered wheelchairs offer more sophisticated seating systems, many users resist transitioning from manual to powered chairs due to concerns about appearing more disabled.

Functional neuromuscular stimulation has shown promise in restoring trunk stability by activating paralyzed muscles, but existing FNS systems have not been effectively integrated with real-time detection of destabilizing events during active wheelchair use. Prior attempts at using inertial measurement units (IMUs) for activity monitoring in wheelchair users have focused primarily on classifying daily activities rather than detecting instability. Mathematical models of wheelchair dynamics have been developed to predict tipping thresholds, but these have not been utilized in active stabilization systems.

The present invention overcomes these limitations by combining real-time inertial event detection with responsive neuromuscular stimulation. The system detects potentially destabilizing situations such as collisions or sharp turns using wheelchair-mounted sensors, then automatically activates appropriate trunk and hip muscles through implanted or surface stimulation to counteract the destabilizing forces. This approach provides active stabilization without restricting voluntary movement, addressing a critical unmet need in wheelchair user safety and independence.

## SUMMARY

The invention provides a threshold-based stabilization system for manual wheelchair users with spinal cord injuries. The system comprises: (1) at least one inertial measurement unit mounted on the wheelchair for detecting destabilizing events; (2) a processing unit implementing detection algorithms that compare sensor data to predetermined thresholds; (3) a stimulation controller that activates appropriate muscle groups when thresholds are exceeded; and (4) implanted or surface electrodes for delivering functional neuromuscular stimulation to the trunk and hip muscles.

Key aspects of the invention include:

1. Real-time detection of destabilizing events through analysis of wheelchair acceleration and angular velocity patterns characteristic of collisions and sharp turns.

2. Subject-specific threshold determination through calibration trials to account for individual differences in wheelchair dynamics and user characteristics.

3. Rapid activation (within 100-400 ms) of trunk and hip muscles through implanted or surface stimulation when destabilizing events are detected.

4. Patterned stimulation of specific muscle groups tailored to counteract different types of destabilizing events (e.g., trunk extensors for forward collisions, lateral stabilizers for turns).

5. Integration with existing implanted pulse generators or surface stimulation systems for practical implementation.

The system demonstrates high detection accuracy (>90%) for both collision and turning events, with clinically significant improvements in trunk stability when stimulation is applied. During collision events, the system reduces forward trunk flexion by an average of 5-15 degrees and decreases recovery time to upright posture. The invention represents a significant advancement over passive restraint systems by providing dynamic, event-responsive stabilization that preserves voluntary movement while reducing fall risk.

## DETAILED DESCRIPTION

The present invention provides a comprehensive system for maintaining trunk stability in manual wheelchair users through real-time detection of destabilizing events and responsive neuromuscular stimulation. The detailed implementation encompasses several key components and methodologies:

1. **Sensor System**: The invention utilizes at least one inertial measurement unit (IMU) mounted on the wheelchair frame, preferably near the rear crossbar. The IMU incorporates a 3-axis accelerometer with minimum ±4g range and a 3-axis gyroscope capable of measuring angular velocities up to 200°/s. Sensor data is sampled at 100Hz or higher to ensure adequate temporal resolution for event detection. Additional sensors may include inclinometers or magnetometers to provide complementary orientation data.

2. **Event Detection Algorithms**: The system implements distinct detection algorithms for different destabilizing events:
   - **Collisions**: Detected when anterior-posterior acceleration exceeds a subject-specific threshold (typically 3.0-4.0g) for a minimum duration (e.g., 50ms).
   - **Turns**: Detected when superior-inferior angular velocity exceeds a threshold (typically 90-120°/s) with directionality indicating left or right turn.
   Thresholds are determined individually for each user through calibration trials, calculated as the mean peak measurement minus two standard deviations from 20 baseline trials.

3. **Stimulation Patterns**: The system activates specific muscle groups in patterns optimized for different destabilizing events:
   - **Collisions**: Simultaneous activation of trunk extensors (erector spinae) and hip extensors (gluteus maximus, hamstrings) to resist forward flexion.
   - **Turns**: Activation of quadratus lumborum on the inside of the turn and hip extensors on the outside to resist lateral displacement.
   Stimulation parameters (pulse amplitude, width, frequency) are customized for each user based on their implanted or surface electrode configuration.

4. **Control System Architecture**: The processing unit implements real-time control using the following components:
   - Signal conditioning filters to remove noise from inertial measurements
   - Threshold comparison modules for event detection
   - Stimulation pattern generators
   - Safety monitoring to prevent overstimulation
   The control algorithms can be implemented in embedded processors or FPGA-based systems for low-latency operation.

5. **User Interface**: The system includes a simple interface allowing users to:
   - Manually disable stimulation after stability is restored
   - Adjust stimulation intensity within safe limits
   - Receive feedback about system status through visual or tactile indicators

6. **Safety Features**: Multiple safety mechanisms are incorporated:
   - Stimulation timeout to prevent prolonged activation
   - Current limiting to prevent tissue damage
   - Redundant sensor validation to reduce false positives
   - Emergency stop function

The system's performance has been validated through extensive testing, demonstrating:
- Collision detection accuracy of 93% with average detection delay of 88ms
- Turn detection accuracy of 93-100% with average delay of 342ms
- Significant reduction in maximum trunk flexion angle during collisions (p<0.05)
- Improved user perception of stability as measured by usability rating scales

Implementation options include integration with existing implanted pulse generators or standalone surface stimulation systems. The invention represents a significant advancement in assistive technology by providing active, responsive stabilization that addresses the limitations of current passive restraint systems while preserving user independence and mobility.