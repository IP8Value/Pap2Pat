# DESCRIPTION

## FIELD

The field of the present invention relates to methods and systems for determining and disseminating earthquake alerts. More specifically, the invention provides a method for identifying and quantifying the enhancement of earthquake risk following large magnitude earthquakes, and a system for generating and disseminating alerts to mitigate potential hazards.

## BACKGROUND

The number of high magnitude earthquakes has been increasing in recent years, with large events often clustered in time. Scientific observations have revealed that large earthquakes can trigger subsequent earthquakes at teleseismic distances, sometimes days after the initial event. Traditional earthquake risk assessment models assume that earthquakes are independent in space and time, ignoring the potential for triggering. This assumption can lead to underestimation of earthquake risks and inadequate preparedness measures.

There is a need for a method and system that can accurately determine the enhancement of earthquake risk following large magnitude earthquakes and provide timely alerts to mitigate potential hazards. The present invention addresses this need by providing a method for identifying and quantifying the enhanced risk of subsequent earthquakes and a system for generating and disseminating earthquake alerts.

## SUMMARY

The present invention provides a method for determining the enhancement of earthquake risk following large magnitude earthquakes. The method involves analyzing the distribution of subsequent earthquakes in relation to the source event, calculating relative rates of triggered earthquakes, and comparing these rates to historical baseline rates. The method further includes identifying statistically significant increases in earthquake rates and determining the spatial and temporal patterns of triggering.

The invention also provides a method for generating and disseminating earthquake alerts based on the identified risk enhancement. The method involves generating alerts for regions with a high probability of experiencing triggered earthquakes and disseminating these alerts to relevant authorities and the public.

Additionally, the invention provides a system for providing earthquake alerts. The system includes a data processing unit configured to analyze earthquake data, a risk assessment module for determining the enhancement of earthquake risk, and an alert generation module for generating and disseminating alerts. The system is designed to operate in a generalized computer environment and can be integrated with existing earthquake monitoring systems.

## DETAILED DESCRIPTION

### I. Terms

For the purposes of this disclosure, the following terms and definitions apply:

- **Source Event**: A large magnitude earthquake (≥M6.5) that has the potential to trigger subsequent earthquakes.
- **Triggered Event**: An earthquake (≥M5.0) that occurs within a specified time and distance from a source event.
- **Relative Rate**: The ratio of the observed rate of triggered earthquakes to the historical baseline rate.
- **Baseline Rate**: The average rate of earthquakes over a long-term period, used as a reference for comparison.
- **Control Group**: A set of time periods without a source event, used to establish the baseline rate.
- **P-value**: A statistical measure indicating the probability that the observed data could have occurred by random chance under the null hypothesis.
- **Aftershock Zone**: The region surrounding a source event where aftershocks are expected to occur, typically defined as 3 times the rupture length of the fault.
- **Antipodal Region**: The region on the Earth's surface that is diametrically opposite to the location of the source event.

### II. Method for Determining Enhancement of Earthquake Risk

The method for determining the enhancement of earthquake risk following large magnitude earthquakes involves the following steps:

1. **Data Collection**: Collect earthquake data from reliable sources, such as the United States Geological Survey (USGS) archives, for all earthquakes with magnitudes ≥M5.0 over a specified time period (e.g., 1973 to 2016).

2. **Aftershock Filtering**: Remove known clustering processes from the data using a standard declustering technique, such as the windowing method applied by Gardner and Knopoff. This step ensures that the data follow a Poisson distribution, which is essential for subsequent statistical analysis.

3. **Test Set Creation**: Create test sets of potential source events based on specific magnitude ranges (e.g., M6.0, M6.5, M7.0, M7.5, ≥M8.0). Each test set includes earthquakes that are greater than or equal to the specified magnitude but less than the next higher magnitude.

4. **Observation Periods**: For each earthquake in a test set, define a three-day observation period starting from the time of the source event. Search the archived data for any earthquake ≥M5.0 that occurred within the next three days.

5. **Arc Distance Calculation**: Compute the arc distance from the source event to each subsequent earthquake found in the three-day observation period. Bin the data at one-degree intervals and then aggregate the data into ten-degree bins, offset every five degrees.

6. **Control Group Creation**: For each member of the test set, create a control group of 5,355 observation periods that do not overlap the test event. Use the latitude and longitude of the test event as the local origin to generate a histogram of the observed counts for each member of the control group.

7. **Baseline Rate Calculation**: Add the binned counts of the control group to define a baseline count of seismic activity as seen from the frame of reference of the given member of the test set. Divide the total count by the size of the control group to obtain the expected numbers for a single three-day period.

8. **Relative Rate Calculation**: Calculate the relative rate of triggered earthquakes by dividing the observed counts by the baseline counts for each ten-degree bin.

9. **Statistical Analysis**: Calculate p-values for the relative rates using the binomial distribution. Identify statistically significant increases in earthquake rates by comparing the p-values to a threshold (e.g., p < 0.05).

10. **Pattern Identification**: Analyze the spatial and temporal patterns of triggering by examining the relative rates and p-values for different magnitude ranges and time lags. Identify regions with a high probability of experiencing triggered earthquakes.

### III. Method for Generating and Disseminating an Earthquake Alert

The method for generating and disseminating earthquake alerts based on the identified risk enhancement involves the following steps:

1. **Alert Criteria**: Define criteria for generating alerts based on the relative rates and p-values. For example, generate alerts for regions with relative rates >1 and p-values < 0.05.

2. **Alert Generation**: Generate alerts for regions with a high probability of experiencing triggered earthquakes. The alerts should include information such as the location, magnitude, and expected duration of the increased risk.

3. **Dissemination Channels**: Disseminate the alerts through multiple channels, including but not limited to:
   - **Government Agencies**: Notify relevant government agencies and emergency management organizations.
   - **Public Media**: Release alerts through television, radio, and online news platforms.
   - **Mobile Applications**: Send push notifications to mobile devices via dedicated earthquake alert applications.
   - **Social Media**: Post alerts on social media platforms to reach a broader audience.

4. **Public Awareness**: Provide educational materials and resources to the public to help them understand the risks and take appropriate safety measures.

### IV. System for Providing Earthquake Alerts

The system for providing earthquake alerts includes the following components:

1. **Data Processing Unit**: A data processing unit configured to collect and analyze earthquake data. The unit should be capable of filtering out aftershocks and performing statistical analysis to identify risk enhancement.

2. **Risk Assessment Module**: A risk assessment module for determining the enhancement of earthquake risk. The module should calculate relative rates and p-values and identify regions with a high probability of experiencing triggered earthquakes.

3. **Alert Generation Module**: An alert generation module for generating and disseminating earthquake alerts. The module should be capable of generating alerts based on predefined criteria and disseminating them through multiple channels.

4. **User Interface**: A user interface for accessing and managing the system. The interface should provide real-time updates on earthquake risk and allow users to customize alert preferences.

5. **Communication Network**: A communication network for transmitting alerts to relevant authorities and the public. The network should support multiple communication channels, including but not limited to government agencies, public media, mobile applications, and social media.

### V. A Generalized Computer Environment

The system for providing earthquake alerts operates in a generalized computer environment, which includes the following components:

1. **Hardware Components**:
   - **Central Processing Unit (CPU)**: A powerful CPU for processing large datasets and performing complex calculations.
   - **Memory**: Sufficient memory to store earthquake data and intermediate results.
   - **Storage**: High-capacity storage for archiving earthquake data and system logs.
   - **Network Interface**: A network interface for connecting to the internet and other communication networks.

2. **Software Components**:
   - **Operating System**: A robust operating system capable of supporting the system's operations.
   - **Database Management System (DBMS)**: A DBMS for storing and managing earthquake data.
   - **Statistical Analysis Software**: Software for performing statistical analysis, such as R or Python.
   - **Web Server**: A web server for hosting the user interface and providing access to the system.
   - **Mobile Application Development Tools**: Tools for developing and deploying mobile applications for alert dissemination.

3. **Security Measures**:
   - **Data Encryption**: Encryption of sensitive data to ensure confidentiality.
   - **Access Control**: Implementation of access control measures to restrict unauthorized access to the system.
   - **Backup and Recovery**: Regular backup of data and implementation of recovery procedures to ensure data integrity.

### VI. General Considerations

1. **Scalability**: The system should be scalable to handle increasing amounts of data and support a growing user base.
2. **Reliability**: The system should be reliable and capable of operating continuously without downtime.
3. **Accuracy**: The system should provide accurate and timely alerts to ensure the safety of the public.
4. **Compliance**: The system should comply with relevant regulations and standards for earthquake monitoring and alert systems.
5. **User-Friendly**: The user interface should be intuitive and easy to use, allowing users to quickly access and understand the information provided.
6. **Continuous Improvement**: The system should be regularly updated and improved based on user feedback and advancements in technology.

By implementing the method and system described herein, the present invention aims to enhance earthquake risk assessment and provide timely alerts to mitigate potential hazards, ultimately contributing to the safety and well-being of communities at risk.