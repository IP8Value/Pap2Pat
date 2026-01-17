# DESCRIPTION

## BACKGROUND

In the realm of videoconferencing, maintaining the natural flow of conversation and the projection of attention cues are critical for effective communication. Traditional videoconferencing systems often fail to replicate the spatial and non-verbal cues that are essential in face-to-face interactions. This disruption is particularly pronounced in hub-and-satellite meetings, where one participant (the satellite) is remotely connected to a group of collocated participants (the hub). The lack of a shared physical environment and the inability to accurately reproduce gaze, body orientation, and pointing gestures contribute to reduced conversational engagement and a diminished sense of presence.

One of the key challenges in videoconferencing is the "newscaster effect," where the satellite participant appears to be looking at everyone and no one simultaneously, leading to a lack of mutual eye contact. Another issue is the "skip-over effect," where the satellite participant is inadvertently overlooked during the conversation. To address these issues, researchers have explored various methods to enhance the projection and awareness of attention in videoconferencing systems.

Kinetic proxies, which are physical devices that can move in response to the satellite participant's actions, have shown promise in mitigating these problems. These proxies can swivel, tilt, or otherwise move to indicate the satellite's focus of attention, thereby improving the hub participants' awareness of the satellite's gaze and engagement. The present invention builds upon this concept by introducing a kinetic proxy that can be controlled both explicitly and implicitly, offering a more nuanced and effective means of projecting attention in videoconferencing environments.

## SUMMARY

The present invention relates to a kinetic proxy system for enhancing attention projection and awareness in videoconferencing, particularly in hub-and-satellite meetings. The kinetic proxy comprises a display screen, a camera, a speaker, and a microphone, all integrated into a motorized turntable. The turntable allows the display screen to swivel horizontally, enabling the satellite participant to direct their attention towards specific hub participants or objects in the meeting room.

The kinetic proxy can be controlled in two primary modes: explicit control and implicit control. In explicit control mode, the satellite participant uses a mouse cursor to select the location they wish to aim the proxy towards. The position of the proxy is directly linked to the position of the mouse cursor over the panorama view of the hub's workspace. In implicit control mode, the proxy screen swivels in response to the satellite participant's head motion, which is automatically tracked using a webcam-based software head tracker.

The invention aims to improve conversational engagement, reduce the newscaster effect, and mitigate the skip-over effect by providing a more natural and intuitive means of projecting attention. The kinetic proxy enhances the hub participants' ability to perceive the satellite's focus of attention, leading to more effective and engaging interactions.

## DETAILED DESCRIPTION

### Overview of the Kinetic Proxy System

The kinetic proxy system is designed to enhance the projection and awareness of attention in videoconferencing, particularly in hub-and-satellite meetings. The system comprises a display screen, a camera, a speaker, and a microphone, all integrated into a motorized turntable. The turntable allows the display screen to swivel horizontally, enabling the satellite participant to direct their attention towards specific hub participants or objects in the meeting room.

### Components of the Kinetic Proxy

1. **Display Screen**: A 12-inch Tablet PC is used to display head-and-shoulders video of the satellite participant. The tablet is mounted in a portrait orientation to approximate the height of a seated hub participant.

2. **Camera**: A fixed-position Axis 212 wide-angle camera captures a panoramic view of the hub's workspace. This view is displayed across the entire width of the satellite's 30-inch monitor, allowing the satellite to see all hub participants and their positions around the table.

3. **Speaker and Microphone**: The kinetic proxy includes a videoconference speakerphone that ensures clear audio communication between the satellite and hub participants.

4. **Motorized Turntable**: The turntable is capable of rotating the display screen within ±90°, allowing the satellite participant to directly face any of the hub participants in the room. The turntable is remotely operated via a USB cable.

### Control Mechanisms

#### Explicit Control

In explicit control mode, the satellite participant uses a mouse cursor to select the location they wish to aim the proxy towards. The position of the proxy is directly linked to the position of the mouse cursor over the panorama view of the hub's workspace. The client program updates the desired "go-to" position approximately 30 times per second, ensuring smooth and responsive movement of the proxy.

#### Implicit Control

In implicit control mode, the proxy screen swivels in response to the satellite participant's head motion. A webcam-based software head tracker is used to track the horizontal component of the satellite's head rotation. The head tracker updates the proxy position approximately 30 times per second, mirroring the satellite's head movements in real-time.

### Benefits of the Kinetic Proxy System

1. **Improved Conversational Engagement**: The kinetic proxy enhances the hub participants' ability to perceive the satellite's focus of attention, leading to more effective and engaging conversations. The swiveling motion of the display screen provides a clear indication of the satellite's attention, reducing the newscaster effect and promoting more natural interactions.

2. **Reduced Skip-Over Effect**: By providing a more natural means of projecting attention, the kinetic proxy helps to mitigate the skip-over effect, where the satellite participant is inadvertently overlooked during the conversation. The hub participants are more likely to include the satellite in the discussion when they can clearly see where the satellite is looking.

3. **Enhanced Sense of Presence**: The kinetic proxy contributes to a greater sense of presence for the satellite participant. The physical movement of the proxy helps to create a more immersive and engaging experience, making the satellite feel more "present" in the meeting.

### Implementation and Usage

The kinetic proxy system is designed to be easy to set up and use. The turntable and display screen are mounted on a lightweight frame constructed of ¼-inch sheet acrylic, which minimizes visual obstructions and positions the display at eye level to the seated hub participants. The hub and satellite participants are evenly distributed around a round conference table, ensuring that the kinetic proxy can effectively reach all participants.

The satellite participant controls the kinetic proxy using either explicit or implicit control mechanisms, depending on their preference. The system is compatible with standard videoconferencing software and can be integrated into existing meeting setups with minimal modifications.

### Experimental Validation

To validate the effectiveness of the kinetic proxy system, a laboratory study was conducted involving multiple groups of participants. The study compared the performance of the kinetic proxy under stationary, explicit control, and implicit control conditions. Key metrics such as speaking time, speech energy, speech segment length, and turn-taking were measured to assess the impact of the kinetic proxy on conversational engagement and attention awareness.

The results of the study demonstrated that the kinetic proxy significantly improved conversational engagement and reduced the newscaster and skip-over effects. Participants in the kinetic conditions spoke for a larger percentage of time, with higher speech energy and longer speech segments, indicating a higher level of engagement. The kinetic proxy also facilitated more accurate responses to deictic prompts, with the intended person responding correctly in 100% of the instances in both the explicit and implicit conditions.

### Future Directions

While the kinetic proxy system has shown promising results, there are several areas for future improvement and exploration. One potential area is the development of more advanced control mechanisms that can better mimic natural head and body movements. Additionally, the use of convex displays or physical pointers could be explored to provide a wider range of attention projection while avoiding the exclusion of turning away from participants.

The kinetic proxy system represents a significant step forward in enhancing the projection and awareness of attention in videoconferencing. By providing a more natural and intuitive means of projecting attention, the system has the potential to significantly improve the effectiveness and engagement of remote collaboration and communication.