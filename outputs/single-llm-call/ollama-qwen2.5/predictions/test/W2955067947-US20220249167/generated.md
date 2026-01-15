### Overview

The study aims to identify optimal targets (TZs) for resection and disconnection surgeries in epilepsy patients. It employs a combination of patient-specific network models and simulations using The Virtual Brain (TVB). The process involves dividing the brain into modules, identifying target nodes/edges, and evaluating their effectiveness and safety through seizure propagation and normal brain function assessments.

### Effectiveness Evaluation

To evaluate the effectiveness of derived TZs, patient-specific network models were constructed using TVB. The Epileptor model was employed to simulate seizure propagation characteristics before and after eliminating target nodes or edges. The suppression ratio of seizure propagation was quantified to compare the removal effect of each TZ, ensuring that the selected TZs effectively reduce seizure spread.

### Safety Evaluation

To assess the impact on normal brain function, a stimulation paradigm was used to quantify information transmission capacity through spatiotemporal properties of the network's response. Eight resting state (RS) networks were tested, and specific cortical regions were stimulated to observe response signals in all brain nodes. The similarity coefficient between pre- and post-surgery activation patterns was calculated, with values below 0.75 indicating a high risk of functional impairment.

### Iterative Feedback Process

The process involves iterative feedback to refine TZs. If the initial TZs pose a high risk (similarity coefficient < 0.75), critical nodes are identified and designated as inoperable zones. The modularity analysis is repeated, generating new TZs. These new TZs are then re-evaluated for effectiveness and safety until a set of TZs meeting both criteria is obtained.

### Clinical Relevance

The study's approach provides multiple intervention options based on the location of the epileptogenic zone (EZ) and inoperable zones. By simulating the worst-case scenario, it ensures that the selected TZs are robust against seizure propagation while minimizing the risk to normal brain function, offering a valuable tool for surgical planning in epilepsy treatment.

### Conclusion

This method offers a systematic approach to identify optimal targets for resection and disconnection surgeries in epilepsy patients. By integrating patient-specific network models with advanced simulations, it ensures that the selected TZs are both effective in reducing seizures and safe in preserving normal brain function, ultimately enhancing the outcomes of surgical interventions.