# DESCRIPTION

## BACKGROUND

Proprioceptive signals from peripheral mechanoreceptors play a critical role in motor control, forming the foundation for movement coordination, postural control, and body awareness. Proprioceptive deficits can severely impair motor control and are common in neurological conditions such as stroke and Parkinson’s disease. Somatosensory-based interventions have been shown to improve proprioceptive function, leading to enhancements in motor function. Proprioceptive training, which focuses on proprioceptive afferent signals, has been demonstrated to enhance wrist proprioceptive acuity in both healthy individuals and those with neurological conditions. Additionally, visuomotor training, which involves making increasingly precise, small-amplitude wrist movements, has been shown to challenge the proprioceptive system and improve proprioceptive function.

Short- and long-term neuroplastic changes in cortical areas have been reported as a result of proprioceptive-focused sensorimotor learning. These changes include reductions in the latency of somatosensory evoked potentials (SEPs) and functional reorganization within the primary sensorimotor cortex and supplementary motor area. Furthermore, numerous studies have established the concept of interlimb transfer of motor learning, where the untrained contralateral limb exhibits signs of motor learning without direct practice. However, the transfer of somatosensory learning to the contralateral limb has not been systematically evaluated.

This invention addresses the gap in understanding the transfer of proprioceptive and motor learning to the contralateral limb by employing a brief, single-session visuomotor training on the right wrist and evaluating wrist sensory and motor function bilaterally for three consecutive days. The primary aim is to determine whether sensory and motor learning that occurs in the trained right wrist transfers to the untrained left wrist and whether such learning transfer is retained after 24 hours.

## SUMMARY

The present invention relates to a method and system for enhancing proprioceptive and motor function in a contralateral limb through visuomotor training. The method involves using a robotic wrist exoskeleton to deliver a visuomotor training task to a first limb, specifically the right wrist. The training task requires the participant to make increasingly precise, small-amplitude wrist movements, challenging the proprioceptive system. The system evaluates the proprioceptive acuity and motor performance of both the trained and untrained limbs before and after the training session, as well as 24 hours post-training.

Key aspects of the invention include:
1. **Robotic Wrist Exoskeleton**: A device capable of delivering precise haptic, position, and velocity stimuli to the wrist, integrated with a virtual reality environment for visual feedback.
2. **Visuomotor Training Task**: A task that requires participants to balance a virtual ball on a tiltable table by making precise wrist movements, with increasing difficulty levels.
3. **Evaluation of Proprioceptive Acuity**: A position sense discrimination task using a psychophysical forced-choice paradigm to determine the Just-Noticeable Difference (JND) threshold.
4. **Evaluation of Motor Performance**: A discrete pointing task to assess movement accuracy error (MAE) in an untrained motor task.

The invention provides a scientific basis for applying proprioceptive-based approaches to clinical populations, such as stroke survivors, by demonstrating that proprioceptive learning can transfer to the contralateral homologous limb segment and that untrained motor function can improve over time.

## DETAILED DESCRIPTION

### Robotic Wrist Exoskeleton

The robotic wrist exoskeleton is a three-degree-of-freedom (DOF) device designed to allow the full range of motion in the human wrist, including flexion/extension, adduction/abduction, and forearm supination/pronation. The exoskeleton is fully backdrivable and powered by four brushless motors, enabling the delivery of precise haptic, position, and velocity stimuli at the wrist. The robot accurately encodes the wrist position at a frequency of 200 Hz with a spatial resolution of 0.0075°. The device is integrated with a virtual reality environment, providing visual feedback of the user’s wrist position during the training session.

### Visuomotor Training Task

The visuomotor training task involves participants sitting comfortably with their right forearm resting on the support splint of the wrist robot and holding the handgrip in a relaxed manner. Participants receive visual feedback of their hand position on a monitor. The task requires participants to balance a virtual ball rolling on a tiltable table within the virtual environment. The participant’s wrist position translates to the virtual table’s angle of inclination. The goal is to keep the ball within a target zone by making precise, small-amplitude wrist flexion/extension movements. A trial is completed upon holding the ball within the target zone for 5 seconds. If the trial is completed within 60 seconds, it is considered successful. The task difficulty increases by altering the virtual mechanical properties, such as the virtual mass of the ball, the gain of the ball’s velocity, and the friction coefficients on the virtual table. Participants use a movement range of 10° wrist extension to 40° wrist flexion to complete the training trials. The training session is limited to a maximum of 90 trials or 45 minutes, with a 2-minute break after every 30 trials.

### Evaluation of Proprioceptive Acuity

Proprioceptive acuity is evaluated using a wrist position sense discrimination task. Participants wear opaque goggles and headphones playing white noise to block visual and auditory cues. In each trial, participants must discriminate between two passively presented stimulus positions: a standard stimulus of 15° flexion and a comparison stimulus always greater than 15°. The order of the standard and comparison stimuli presentation is randomized. The robot moves the wrist at a velocity of 6°/s from the start (neutral position) to each stimulus position, holds for 2 seconds, and then moves back to the start. After both positions are presented, participants verbally indicate which position (first or second) is farther from the start position. The subsequent stimulus pair is determined based on the participant’s verbal response using an adaptive psychophysical psi-marginal algorithm. The complete evaluation consists of 30 trials, with a 2-minute break after 15 trials to avoid testing-related fatigue. Based on the participant’s verbal responses, a Just-Noticeable Difference (JND) threshold is determined by fitting the correct response rate and the stimulus difference size using a logistic Weibull function.

### Evaluation of Motor Performance

Motor performance is assessed using a discrete pointing task in the absence of vision. Participants wear opaque goggles and headphones playing white noise to block visual and auditory cues. The wrist robot passively moves the participant’s wrist to 15° flexion from the start (neutral) position, holds it for 2 seconds, and then moves back to the start position. This allows participants to experience the target based on proprioceptive information. Subsequently, participants actively move the wrist to the perceived target position and hold it for 2 seconds. The wrist robot records the angular position of the wrist joint during the 2-second hold period. The absolute angular error between the target position (15°) and the final joint position at the end of the pointing movement is computed for each trial. The mean absolute angular error across all trials for each participant is calculated as the Movement Accuracy Error (MAE) to represent a measure of untrained motor performance.

### Data Analysis

Data for all participants are collected bilaterally before training, immediately after training, and 24 hours post-training to determine the influence and retention of right wrist visuomotor training. Data distributions for all variables are examined for normality using the Shapiro-Wilk test. Outliers are defined using the criteria of falling 1.5 times the interquartile range (IQR) below the first quartile or 1.5 times IQR above the third quartile. Paired t-tests are performed on all comparisons for the right and left wrist to determine training-related differences in JND and MAE between the three assessments. The initial significance level is set at p value = 0.05. To account for multiple testing, false discovery rate corrections using the Benjamini-Hochberg procedure are applied. The correlation analysis focusing on how trained motor performance changes over the number of trials uses Spearman Correlation for Cumulative Spatial Error (CSE) and Pearson-Product Correlation for Movement Time (MT).

### Examples

#### Example 1: Enhancing Proprioceptive Acuity and Motor Performance in Healthy Adults

**Participants**: Fifteen healthy right-handed individuals (age: 24.67 ± 4.19 years; 8 males) with no known neurological conditions.

**Procedure**:
1. **Pretest**: Participants complete the handedness questionnaire and undergo pretest assessments of proprioceptive acuity and motor performance bilaterally.
2. **Training Session**: Participants engage in a single session of visuomotor training on the right wrist, completing up to 90 trials or 45 minutes.
3. **Posttest**: Immediately after training, participants undergo posttest assessments of proprioceptive acuity and motor performance bilaterally.
4. **Retention**: 24 hours post-training, participants undergo retention assessments of proprioceptive acuity and motor performance bilaterally.

**Results**:
- **Proprioceptive Acuity**: Mean JND decreased from 1.40° (SD: 0.39°) to 1.03° (SD: 0.27°) in the trained right wrist, with a significant mean relative change (27% decrease, t = 5.11, p < 0.001, d = 1.32). In the untrained left wrist, mean JND decreased from 1.37° (SD: 0.37°) to 0.93° (SD: 0.25°), with a significant mean relative change (32% decrease, t = 6.86, p < 0.001, d = 1.77).
- **Motor Performance**: Mean MAE decreased from 2.50° (SD: 1.00°) to 1.67° (SD: 0.57°) in the trained right wrist, with a significant mean relative change (33% decrease, t = 2.73, p = 0.026, d = 0.73). In the untrained left wrist, mean MAE decreased from 2.31° (SD: 0.67°) to 1.91° (SD: 0.57°), with a non-significant mean relative change (17% decrease, t = 2.20, p = 0.07).

**Conclusion**: The visuomotor training significantly enhanced proprioceptive acuity and motor performance in the trained right wrist, with evidence of contralateral transfer to the untrained left wrist. The improvements in proprioceptive acuity were retained for up to 24 hours, while the gains in motor performance consolidated over the same period.

#### Example 2: Application in Clinical Populations

**Participants**: Stroke survivors with unilateral proprioceptive and motor dysfunction.

**Procedure**:
1. **Pretest**: Participants undergo pretest assessments of proprioceptive acuity and motor performance bilaterally.
2. **Training Session**: Participants engage in a single session of visuomotor training on the less affected right wrist, completing up to 90 trials or 45 minutes.
3. **Posttest**: Immediately after training, participants undergo posttest assessments of proprioceptive acuity and motor performance bilaterally.
4. **Retention**: 24 hours post-training, participants undergo retention assessments of proprioceptive acuity and motor performance bilaterally.

**Expected Results**:
- **Proprioceptive Acuity**: Significant improvements in JND in the trained right wrist, with evidence of contralateral transfer to the more affected left wrist.
- **Motor Performance**: Significant improvements in MAE in the trained right wrist, with evidence of contralateral transfer to the more affected left wrist.

**Conclusion**: The visuomotor training can be effectively applied to clinical populations, such as stroke survivors, to enhance proprioceptive and motor function in the more affected limb by training the less affected limb. Systematic clinical trials are needed to confirm the clinical significance of this approach.

### Conclusion

The invention provides a robust method and system for enhancing proprioceptive and motor function in a contralateral limb through visuomotor training. By demonstrating the transfer of proprioceptive and motor learning to the untrained limb, the invention offers a scientific basis for applying proprioceptive-based approaches to clinical populations, particularly those with unilateral proprioceptive and motor dysfunction. The use of a robotic wrist exoskeleton and a well-designed visuomotor training task ensures precise and reliable evaluations of proprioceptive acuity and motor performance, making this invention a valuable tool in neurological rehabilitation.