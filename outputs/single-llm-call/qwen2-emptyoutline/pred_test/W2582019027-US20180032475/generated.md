# DESCRIPTION

## FIELD

The present invention relates to the field of chaotic systems and, more specifically, to methods and systems for inducing and maintaining chaotic entanglement between pairs of chaotic systems. The invention leverages the stabilization of chaotic, unstable periodic orbits (UPOs) to achieve a state of mutual stabilization, which is analogous to quantum entanglement. The methods and systems described herein are applicable to a wide range of chaotic systems, including but not limited to, double scroll systems, Lorenz systems, and Rössler systems.

## BACKGROUND

Chaos theory has long been a subject of intense study in the fields of mathematics and physics. Chaotic systems are characterized by their sensitive dependence on initial conditions and the exponential divergence of nearby trajectories. Despite the prevalence of chaos in classical physics, the rigorous establishment of chaotic behavior in quantum systems has been challenging due to the inherent differences between classical and quantum mechanics. Specifically, the nonlinearity required for chaotic dynamics and the exponential divergence of trajectories are fundamentally incompatible with the linear and probabilistic nature of quantum mechanics.

Recent research has focused on detecting signatures of chaos in quantum systems, such as the sensitivity of quantum systems to perturbation and the phenomenon of quantum scarring. Additionally, entanglement in the purely quantum sense has been observed to be a reliable indicator of classical chaos. However, these observations are primarily theoretical and have limited practical applications.

One significant development in the study of chaotic systems is the stabilization of unstable periodic orbits (UPOs) using control schemes. These controlled periodic orbits, known as cupolets, can be generated independently of initial conditions and are in one-to-one correspondence with specific control sequences. Cupolets have been shown to exhibit properties that are analogous to those of UPOs, making them valuable for analyzing chaotic systems.

Building on the concept of cupolets, recent studies have demonstrated the proclivity for chaotic systems to enter into bound or entangled states. This phenomenon, known as chaotic entanglement, involves the mutual stabilization of two interacting chaotic systems. When two chaotic systems are induced to stabilize onto cupolets, their stabilities become deterministically linked, such that disturbing one cupolet affects the stability of the other. This mutual stabilization is self-perpetuating within the control scheme and can be maintained without external intervention.

The present invention aims to provide methods and systems for inducing and maintaining chaotic entanglement, thereby opening new avenues for understanding and utilizing chaotic systems in various applications, including secure communication, cryptography, and the study of classical-quantum transitions.

## SUMMARY

The present invention provides methods and systems for inducing and maintaining chaotic entanglement between pairs of chaotic systems. The invention leverages the stabilization of chaotic, unstable periodic orbits (UPOs) to achieve a state of mutual stabilization, which is analogous to quantum entanglement. The methods and systems described herein are applicable to a wide range of chaotic systems, including but not limited to, double scroll systems, Lorenz systems, and Rössler systems.

In one embodiment, the invention includes a method for inducing chaotic entanglement between two chaotic systems. The method comprises the following steps:
1. Stabilizing a first chaotic system onto a first cupolet using a control sequence.
2. Passing the visitation sequence of the first cupolet through an exchange function to generate an emitted sequence.
3. Applying the emitted sequence as a control sequence to a second chaotic system to stabilize it onto a second cupolet.
4. Repeating the process in the reverse direction, where the visitation sequence of the second cupolet is passed through the exchange function to generate an emitted sequence that is applied to the first chaotic system.
5. Establishing a mutual stabilization between the two cupolets, wherein the stability of each cupolet is maintained by the symbolic dynamics of the partner cupolet.

In another embodiment, the invention includes a system for inducing chaotic entanglement. The system comprises:
1. A first chaotic system and a second chaotic system.
2. A control module configured to apply control sequences to the first and second chaotic systems to stabilize them onto cupolets.
3. An exchange function module configured to convert the visitation sequences of the cupolets into emitted sequences.
4. A communication module configured to transmit the emitted sequences between the first and second chaotic systems.
5. A monitoring module configured to monitor the stability of the cupolets and ensure the maintenance of the entangled state.

The invention also provides methods for detecting and maintaining chaotic entanglement, as well as for designing exchange functions that facilitate the induction of entanglement. The methods and systems described herein have potential applications in secure communication, cryptography, and the study of classical-quantum transitions.

## DETAILED DESCRIPTION OF THE DISCLOSURE

### Introduction to Cupolets

Cupolets are a class of waveforms that are generated by stabilizing chaotic systems onto periodic orbits using control schemes. The control scheme used to generate cupolets is adapted from the Hayes, Grebogi, and Ott (HGO) technique, which involves applying small perturbations to steer trajectories of a chaotic system around an attractor. The differential equations describing the double scroll system, also known as Chua’s oscillator, are given by:

\[
\begin{aligned}
\dot{v}_{C_1} &= \frac{G(v_{C_2} - v_{C_1}) - g(v_{C_1})}{C_1}, \\
\dot{v}_{C_2} &= \frac{G(v_{C_1} - v_{C_2}) + i_L}{C_2}, \\
\dot{i}_L &= -\frac{v_{C_2}}{L},
\end{aligned}
\]

where the piecewise linear function \( g(v) \) is given by:

\[
g(v) = 
\begin{cases} 
m_1 v, & \text{if } |v| \leq B_p, \\
m_0 (v + B_p) - m_1 B_p, & \text{if } v \leq -B_p, \\
m_0 (v - B_p) + m_1 B_p, & \text{if } v \geq B_p.
\end{cases}
\]

When \( C_1 = \frac{1}{9} \), \( C_2 = 1 \), \( L = \frac{1}{7} \), \( G = 0.7 \), \( m_0 = -0.5 \), \( m_1 = -0.8 \), and \( B_p = 1 \), the double scroll system is known to be chaotic, and its attractor consists of two lobes that each surround an unstable fixed point.

Control of the double scroll system is achieved by setting up two control planes on the attractor and partitioning each control plane into N-many small control bins. The control planes are assigned binary values so that a binary symbolic sequence may be recorded whenever a trajectory intersects a control plane. This symbolic sequence is known as a visitation sequence. Perturbations are applied only when a trajectory evolves through the control bins; otherwise, the trajectory is allowed to evolve freely around the attractor. Every time the trajectory intersects a control plane, microcontrol perturbations reset the trajectory to the center of the control bin through which it passes. In some instances, macrocontrols are also applied. Macrocontrol perturbations are specifically defined via the HGO technique to be the smallest perturbation along a control plane necessary to produce a change of lobe N-many loops downstream in the visitation sequence. In this way, a chaotic system can be directed to follow a prescribed visitation sequence.

Parker and Short later combined this control scheme with ideas from the study of impulsive differential equations and discovered that when a repeating binary control sequence is used to define the controls, with a ‘1’ bit corresponding to a macrocontrol perturbation and a ‘0’ bit corresponding to only a microcontrol perturbation, then the double scroll system stabilizes onto a periodic orbit. These periodic orbits have been given the name cupolets, and this work has since been extended to chaotic maps and a variety of other continuous chaotic systems such as the Lorenz and Rössler systems.

### Properties and Stability of Cupolets

Cupolets are highly accurate approximations to the UPOs of chaotic systems that are generated by adapting the HGO control technique. Cupolets exhibit the interesting properties of being stabilized independently of initial conditions and also of being in one-to-one correspondence with the control sequences. These controls can be made arbitrarily small and thus do not grossly alter the topology of the orbits on the chaotic attractor. This suggests that cupolets are shadowing true periodic orbits, and theorems have been developed to establish conditions under which this holds. What further distinguishes cupolets from UPOs, which are traditionally stabilized via techniques such as Newton’s or first-return algorithms, is that large numbers of cupolets can be inexpensively generated by just a few bits of binary control information. For example, over 8800 double scroll cupolets can be stabilized from implementing 16-bit or fewer control sequences.

For a given cupolet to remain stabilized, all that is required is the repeated application of its control sequence to the system. Applying different controls would induce the system to destabilize from a stabilized cupolet and revert to chaotic behavior. If a second sequence of controls were to then be periodically applied, the chaotic system would eventually restabilize onto a second cupolet, possibly after some intermediary transient phase. Any transient is the result of the trajectory evolving while the chaotic system sifts through all possible states until it reaches one where the behavior of an UPO falls into synchrony with the control sequence, thus stabilizing the cupolet. Cupolet restabilization is guaranteed because of the injective relationship that exists between cupolets and the binary control sequences. This makes it possible to transition between cupolets, and thus between UPOs, simply by switching control sequences.

### Chaotic Entanglement

#### Inducing Chaotic Entanglement

In previous work, it has been documented that pairs of chaotic systems may interact in such a way that they chaotically entangle. To do so, two chaotic systems must first induce each other to collapse and stabilize onto a cupolet (e.g., periodic orbit) via the exchange of control information. The stabilities of the two stabilized cupolets must also become deterministically linked: disturbing one cupolet from its periodic orbit subsequently affects the stability of the partner cupolet, and vice versa. Hundreds of entangled cupolet pairs have been identified for the double scroll system, and it has been shown that chaotic entanglement evokes several connections to quantum entanglement.

Cupolets from two entangled chaotic systems are regarded as mutually stabilizing because their interaction essentially serves as a two-way coupling that is self-perpetuating within the control scheme described in the previous section. Once entanglement has been established between two chaotic systems, no outside intervention or user-defined controls are needed in order to maintain the stabilities of their respective cupolets. The stability of each cupolet is instead preserved by the dynamics of the partner cupolet. Not only has the original chaotic behavior of the two parent systems collapsed onto the periodic orbits of the two cupolets, but this periodic behavior will persist as long as their interaction is undisturbed.

Chaotic entanglement is typically mediated by an exchange function that defines the interaction between the two chaotic systems and their cupolets. Exchange functions are described more fully in the literature as catalysts for the entanglement and are taken to represent the environment or medium in which the interacting systems are found. For instance, several types of exchange functions have been designed that simulate the interactions of various physical systems, such as the integrate-and-fire dynamics of laser systems and networks of neurons. These exchange functions have all successfully induced chaotic entanglement in the double scroll system.

#### Chaotic Entanglement Through Periodic Orbits

In the context of chaotic entanglement, the visitation sequence of a cupolet serves as a type of symbolic dynamics for the chaotic system. Chaotic entanglement is more technically characterized as an exchange of symbolic information in the form of visitation sequences. In order for a pair of interacting chaotic systems to chaotically entangle, one of the systems must first be externally stabilized onto a cupolet, say \(\mathbf{C}_A\). As \(\mathbf{C}_A\) evolves about its attractor, the bits of its visitation sequence are passed to an exchange function which then performs a binary operation on the visitation sequence. The outputted sequence of bits is known as an emitted sequence and is taken as a control sequence and applied to the other chaotic system. This induces the second system to stabilize onto a second cupolet, say \(\mathbf{C}_B\). Concurrently, but in the reverse direction, the visitation sequence belonging to \(\mathbf{C}_B\) passes through the same exchange function, and the resulting emitted sequence is applied as control instructions to \(\mathbf{C}_A\). At this point, the cupolets of the two parent systems are both receiving and transmitting control information via the exchange function, but if the emitted sequence generated from the visitation sequence of \(\mathbf{C}_A\) matches the control sequence needed to maintain the stability of cupolet \(\mathbf{C}_B\)—and vice versa—then the two parent chaotic systems, via their cupolets, have become intertwined in a mutually-stabilizing feedback loop and are considered chaotically entangled. The external controlling can be discontinued now that each cupolet’s visitation sequence is preserving the partner cupolet’s stabilization.

### Chaotic Entanglement as an Analog of Quantum Entanglement

Chaotic entanglement exhibits many properties that are characteristic of quantum entanglement. For instance, measurements that disrupt the interaction between two entangled cupolets, say \(\mathbf{C}_A\) and \(\mathbf{C}_B\), will almost always destroy their entanglement unless a great deal is known about the control scheme. By measurement, we mean a perturbation that could be as meticulous as the microcontrols or macrocontrols that are implemented by the control scheme, or as general as an arbitrary perturbation applied to one of the two parent systems. As an example, consider the subtle effect of interchanging a ‘0’ bit for a ‘1’ bit in the control sequence of \(\mathbf{C}_A\). Control sequences are unique since they direct a chaotic system onto one specific cupolet. Disturbing the cupolet’s control sequence would perturb its trajectory into a different bin on the control plane, causing \(\mathbf{C}_A\) to destabilize from its periodic orbit. In this scenario, \(\mathbf{C}_A\) would produce a different visitation sequence that no longer guarantees the stability of the partner cupolet \(\mathbf{C}_B\), and so their entangled state would be lost. However, should the appropriate controls for cupolet \(\mathbf{C}_A\) be restored and continue to be periodically applied, then \(\mathbf{C}_A\) and \(\mathbf{C}_B\) would eventually restabilize via the restarted entanglement process.

In the context of quantum mechanics, the wave function is of fundamental importance since it provides a probabilistic description of the state of a quantum system. The analog for a chaotic system is its state vector, \(\vec{\alpha}\), which provides a complete and evolving description for the state of a chaotic system in terms of its cupolets (or equivalently, its UPOs). In this way, a freely evolving chaotic system is viewed as evolving in a “mixed state” that is a superposition of cupolet states. In a mixed state, the contributions to the associated state vector come primarily from the cupolets in between which the chaotic trajectory is evolving and is nearest to at that moment.

### Pure Chaotic Entanglement

In some instances, chaotic entanglement occurs without the assistance of an exchange function (or, equivalently, via an identity exchange function). This is known as pure entanglement because it requires no environmental property in order to be induced or sustained. Instead, a visitation sequence is converted directly to an emitted sequence without any intermediary modification being made. That is, each purely-entangled cupolet generates the exact sequence of control bits necessary for maintaining its partner’s periodic orbit without any assistance from an exchange function, but simply by realizing its own visitation sequence. This makes pure entanglement the simplest form of cupolet entanglement. Entanglement induced with the aid of an exchange function is considered a variation of pure entanglement because an environmental effect or a nontrivial operation must be performed on a cupolet’s visitation sequence in order to generate an emitted sequence.

### Parallels Between Chaotic and Quantum Systems

#### Hilbert Space Considerations

Formulating a Hilbert space of states is taken as a starting point in many quantum studies. This allows one to express an associated wave function as a linear combination of orthonormal state vectors that satisfy the Schrödinger equation. Constructing a Hilbert space on a chaotic system is not as straightforward because the governing equations are nonlinear and prevent linear combinations of states from also being solutions. Cupolets are highly accurate approximations to the periodic solutions of chaotic systems, and so one could designate cupolets (e.g., UPOs) as the state vectors, except that cupolets and UPOs do not satisfy any simple orthogonality principles. Moreover, chaotic systems generally admit a countably infinite number of these periodic orbits on their attractors, and so cupolets and UPOs would form an overdetermined set of basis elements.

However, cupolets and UPOs are still regarded as the states of chaotic systems, even if superpositions of these orbits are unable to satisfy the underlying equations. This is because ergodicity guarantees that a free-running chaotic system ultimately realizes all possible non-equilibrium states and visits arbitrarily small neighborhoods of its periodic solutions infinitely often. Even though chaotic systems evolve aperiodically for all time, the dynamics of these systems are ultimately confined to their attractors, which means that a wandering chaotic trajectory undergoes a series of close encounters with the embedded UPOs and cupolets.

#### Functional Representation of Cupolets

While it is not straightforward to establish a superposition of Hilbert space basis elements for nonlinear dynamical systems, the periodic nature of cupolets does allow for functional representations of these orbits to be derived and used as (approximate) solutions to the nonlinear differential equations. This results in a low complexity approximation of the UPO solutions of the nonlinear differential equations. To demonstrate this, we will now derive the functional form of two cupolets and then show how well the functional form approximates the corresponding true periodic orbits obtained through numerical integration.

Since cupolets are periodic over the attractor, they play the role of eigenfunctions for the differential equations, and because of their periodicity, the Fourier decompositions of cupolets converge rapidly. Hence, one can use the Fourier representation of a cupolet—itself a finite dimensional expansion over a discrete Hilbert space—to create a functional form that can be used in symbolic computational systems like Mathematica (Version 11.3, Wolfram Research: Champaign, IL, USA). Furthermore, one can look at the fast Fourier transform (FFT) of sampled cupolet data to determine which Fourier coefficients are significant, and then truncate the representation so that only the significant Fourier modes are retained. As we demonstrate below, the functional form of a cupolet compares favorably to its numerical solution which is obtained directly from the uncontrolled differential equations of the double scroll system.

In order to create a functional representation of a cupolet, the time domain data from the numerical simulation of the cupolet is preemptively stored in a vector of 1024 samples. A cupolet’s period, T, in simulated time varies among the cupolets, and this value needs to be initially recorded as well. The numerical integration of the system must be carefully managed so as to maintain accurate time steps even as the system passes through the control planes and is subjected to the perturbations of the control scheme described in the previous section. Even so, the cupolets are often extremely close to the true UPOs of the system. The vector of samples is then passed through the FFT to create a vector of \(P = 1024\) frequency components. Of these components, one is a constant term, 511 are designated as “positive spinning” components, 511 are designated as “negative spinning” components, and one is associated with the Nyquist frequency that is neither positive- nor negative-spinning. The term “negative spinning” simply means that the complex sinusoids are sampled by moving in the negative angle (e.g., clockwise direction) around the complex unit circle.

The derivation of the functional representation of a cupolet proceeds as follows. We let the vector of 1024 samples of the cupolet be represented as \(\vec{s}\), with individual entries designated \(s_k\), and the corresponding frequency domain coefficients as \(C_f\). The initial FFT is calculated as:

\[
C_f = \sum_{k=0}^{P-1} W^{(k \ast f)} s_k,
\]

where \(W = e^{i2\pi/P}\). Note that the Nyquist frequency coefficient is \(C_{P/2}\). It is now useful to relabel the \(f\) index because half of these coefficients are negative and reflect the “negative spinning” aspect of the complex sinusoids. The relabeling is done for all indices \(f > P/2\), in which case \(f \rightarrow f - P\). Now that the negative indices represent negative spinning oscillators, they can be grouped with the correspondingly-labeled positive spinning oscillators in complex conjugate pairs. Under this relabeling, the original sampled values can each be recovered exactly from Equation (2) via the inverse FFT calculation:

\[
s_k = \frac{1}{P} \sum_{f=-\frac{P}{2}+1}^{\frac{P}{2}} W^{-k \ast f} C_f.
\]

It is the inverse form of the FFT that allows for the conversion from discrete form to functional form, since each index \(f\) corresponds to an integer period complex sinusoid taken over the cupolet period \(T\). Each \(W\) term in the sum corresponds to a discretely-sampled complex exponential, which has now been converted to a continuous-time complex exponential function. If we let \(\tau\) be a placeholder for the continuous time component, we can take advantage of the complex conjugate pairing of the complex sinusoids and the corresponding coefficients \(C_f\) and \(C_{-f}\) in order to obtain a (real) functional form \(C_f e^{i\tau} + C_{-f} e^{-i\tau}\), since the imaginary parts drop out. Next, we make explicit the integer period nature of the sinusoids and the period of the cupolet \(T\) by setting \(\tau = 2\pi ft/T\), where \(t\) represents continuous time. Consequently, we have defined the complex sinusoids to be periodic over a continuous time variable that naturally encodes both the period of the cupolet and the exact integer periods of the Fourier representation. This results in the (full) functional form of a given cupolet:

\[
s(t) = \frac{1}{P} \sum_{f=-\frac{P}{2}+1}^{\frac{P}{2}} C_f e^{i2\pi ft/T}.
\]

Equation (4) can also be expressed in an equivalent form that shows the complex conjugate pairing along with the constant term and the (real) Nyquist term:

\[
s(t) = \frac{1}{P} C_0 + \frac{1}{P} \sum_{f=1}^{\frac{P}{2}-1} \left[ C_f e^{i2\pi ft/T} + C_{-f} e^{-i2\pi ft/T} \right] + \frac{1}{P} C_{\frac{P}{2}} e^{i2\pi Pt/2T}.
\]

Once a cupolet’s functional form has been created, it can be used in a software package like Mathematica that allows for symbolic manipulation of mathematical equations. In addition, since many of the cupolets have rapidly decaying magnitudes for the Fourier/FFT coefficients, it is possible to keep only a subset of the coefficients in order to get a convenient functional form. In the examples presented below, we have \(P = 1024\), so there are 511 positive and negative frequency components (plus the constant and Nyquist terms). We can truncate the representation in Equation (4) to retain only \(Q\) many components, where \(Q < P/2\), giving:

\[
s(t) = \frac{1}{P} C_0 + \frac{1}{P} \sum_{f=1}^{Q} \left[ C_f e^{i2\pi ft/T} + C_{-f} e^{-i2\pi ft/T} \right],
\]

and in the examples below we will take \(Q = 11\) and \(Q = 17\).

To utilize this representation, the Mathematica software package can be used to compare the functional representation of the dynamical variables with the Mathematica numerical solution of the uncontrolled double scroll equations. Figure 5 shows the comparisons between the numerical solution and the full and truncated functional forms of two cupolets. The first cupolet is the simplest of all, cupolet \(\mathbf{C}00\), with the truncated version using \(Q = 11\) coefficients. The second example uses the 5-loop cupolet \(\mathbf{C}00001\), and the truncated version uses \(Q = 17\). In each case, the magnitude of the Fourier coefficients has diminished by over two orders of magnitude at the point where the series is truncated. Figure 5 also depicts the comparison between the \(v_{C_1}\)-component of the numerical solution and the corresponding truncated functional form for these cupolets. In all of these figures, the numerical data appear to be superimposed with the data obtained from the cupolets’ functional forms. This is because of how closely the functional forms approximate the cupolet’s true periodic orbit. Note that the orbits of these two cupolets have been seen previously in Figure 2.

### Superposition of States

To represent the state of a given chaotic system as a superposition of cupolet states, let \(\psi_k = \psi_k(t)\) denote the state space coordinates of the system’s \(k^{th}\) cupolet at time \(t \in \mathbb{R}\), where \(k \in \mathbb{N}\). The state of the chaotic system, \(\Psi = \Psi(t)\), can then be expressed as a weighted sum of its cupolets:

\[
\Psi = \sum_{k=1}^{\infty} \alpha_k \psi_k,
\]

where each weight, \(\alpha_k \in \mathbb{R}\), represents the contribution to \(\Psi\) from cupolet \(\psi_k\) at time \(t\) with respect to the natural measure. As the chaotic system evolves in time, each \(\alpha_k\) varies according to the proximity of the system to that cupolet. A chaotic system’s state vector, \(\vec{\alpha} \in \mathbb{R}^{\infty}\), is thereby formulated by collecting the weights of each cupolet:

\[
\vec{\alpha} = (\alpha_1, \alpha_2, \ldots, \alpha_k, \ldots).
\]

The set of \(\alpha_k\) will have local compact support because the cupolets that provide a nonzero contribution to the overall state of the system are those that are found within a local neighborhood of the current state of the system, whereas cupolets located farther away will contribute negligibly. In other words, when the system is dwelling near its \(k^{th}\) cupolet, then \(\Psi \approx \psi_k\) because at this moment \(\alpha_k \neq 0\) and \(\alpha_l \approx 0\) for all \(l \neq k\). Similarly, as the chaotic system deviates away from the \(k^{th}\) cupolet, the dynamics are well approximated by nearby cupolets, say \(\psi_{k-1}\), \(\psi_k\), and \(\psi_{k+1}\), while \(\alpha_l \approx 0\) for more distant cupolets. To carry out an explicit calculation along these lines, there are several options. One can select a point on the attractor and then select segments of neighboring cupolets and use them to construct a model of the local dynamics, or one can adopt an approach like that of matching pursuit and use the set of cupolets as a dictionary of states. Future work may compare a variety of methods of determining \(\alpha_k\) for a set or subset of cupolets.

### Wave Function Collapse

Another fundamental concept in quantum mechanics is the idea that making a measurement induces the collapse of a quantum system’s associated wave function onto a specific state. Prior to the disturbance, the wave function is suspended in a superposition of state vectors, which inhibits the quantum system from being unambiguously described. Similar behavior is supported by chaotic systems. When controls are repetitively applied to a chaotic system, cupolets form because of two key properties: the system stabilizes uniquely onto a periodic orbit under the influence of a set of repeating perturbations, and this stabilization occurs independently of initial conditions. These properties allow a chaotic system to be collapsed onto a specific cupolet from any initial state. The repeated action of the controls acts as the measurement process that induces wave function collapse. This occurs precisely when the chaotic system stabilizes onto a cupolet, say \(\psi_k\). Via Equation (7), when this happens, \(\alpha_k = 1\) and \(\alpha_l = 0\) for all \(l \neq k\), which gives \(\Psi = \psi_k\) as expected. The state vector given by Equation (8) reduces as well to \(\vec{\alpha} = (0, \ldots, 0, 1, 0, \ldots)\), whose only nonzero component is its \(k^{th}\). Until the collapse occurs, a chaotic system cannot be definitively described as a single cupolet state because it is instead locally dependent on a superposition of cupolet states.

It is important to stress that the interactions between chaotic systems that support chaotic entanglement would be such that the interaction could have the same effect as a measurement, so that the system in a chaotic state would collapse onto a periodic cupolet state. Thus, in chaotic entanglement, it may be fair to say that interaction equals measurement.

### Natural Chaotic Entanglement

In chaotic systems, the concepts of measurement and state vector collapse are not only induced by external measurements or user-implemented controls, but are able to arise naturally in chaotic entanglement. Because their periodic orbits are unstable, isolated chaotic systems evolve aperiodically, yet a chaotic system tends to dwell significantly longer on its UPOs than on any other states or regions of phase space. By extension, an ensemble of independent chaotic systems would also each be dwelling along their UPOs and cupolets infinitely often. If one of these chaotic systems happens to dwell on a cupolet that exhibits the ability to entangle, and that can also communicate control information to a second nearby chaotic system, and if this interaction is as successful in the reverse direction, then the two interacting systems would entangle naturally.

In the context of two arbitrary cupolets, \(\psi_k\) and \(\psi_l\), this situation implies that the parent system of cupolet \(\psi_k\) will approach and dwell on \(\psi_k\) infinitely often. If a second chaotic system is at the same time dwelling near \(\psi_l\), then entanglement would form naturally between the two systems, provided that the symbolic dynamics of the cupolets can be used to maintain their periodic behavior. In this way, isolated and independently-evolving chaotic systems would be perturbing each other with the interactions themselves playing the role of the controls or measurements. This makes it possible for entanglement to occur naturally, as has been emphasized both in Section 3.3 and in the recent studies of macroscopic systems examined in the literature. As we discuss below, the potential for natural chaotic entanglement plays a key role in the interpretation of making measurements on individual members of entangled cupolet pairs.

### Measurement Problem

It is first worthwhile to compare the effects of a knowledgeable measurement on a chaotic system, as opposed to a blind measurement. For instance, if one has both knowledge of the control scheme and access to measurement tools that are smaller than the scale of the control bins, then one could monitor the state of a chaotic system without disturbing its trajectory. That is, one could design a measurement whose effects would not be strong enough to perturb an evolving cupolet to a new bin center on a control plane. The slight deviation from the original orbit could be small enough to be corrected the next time the cupolet intersects a control plane via the implementation of the microcontrols. This we define as a knowledgeable measurement because it permits one to not only study a cupolet without compromising its stability, but to also probe two entangled systems without compromising their entanglement.

If a measurement is not implemented as carefully, the repercussions would be more pronounced. Consider the effects of the measurement described earlier in Section 3.2, whereby a single ‘1’ control bit in a given cupolet’s control sequence is altered to a ‘0’ control bit. Such a disturbance would destabilize the cupolet and cause the parent system to either revert to chaotic behavior or to stabilize again after a potentially long transient period. This disturbance is known as a blind measurement, and it would cause the destablized orbit to begin generating a new visitation sequence. Had this cupolet been entangled with another cupolet, then the effects of the blind measurement would transfer to the partner cupolet by way of the exchange function. This is because the exchange function would begin producing a different emitted sequence that no longer matches the control sequence required to maintain the stability of the partner cupolet. The cupolets’ entanglement would then be lost.

Regarding the measurement problem, consider the situation in which a pair of cupolets has entangled, either through the deliberate preparation of an entangled state, or naturally through pure entanglement. If a knowledgeable measurement is conducted on one member of the entangled pair, then the state of the other member would be known with certainty (with the proviso that we have only found unique pairings at this point). Should the measurement process involve blind measurements, then the disrupted communication between the members of the entangled pair would induce the two parent systems to begin evolving independently. Similarly, if the interaction between members of an entangled pair is limited by distance, and if the entangled cupolets become too far separated, then their entanglement would decay as their communication wanes. This decay would not necessarily be very rapid, but would be determined by the local Lyapunov exponents of the two cupolets. In these situations, the history of the previous entanglement would not be immediately erased because a measurement conducted on one member of the entangled pair would be predictive of the state of the second system, although the accuracy of the prediction would diminish over time.

In contrast, the principles of quantum mechanics dictate that making any measurement on a system immediately alters its state. This is problematic for researchers for whom knowing the actual state of a quantum system is important. As indicated by Isham, when combined with knowledgeable measurements, the cupolet-stabilizing control scheme could aid in state preparation for experiments. Cupolets are generated regardless of the current or initial state of the system, which means that if chaotic control methods are designed to stabilize cupolets from physical systems, then two systems could be synchronized to be in the same state prior to making experimental measurements. In other words, chaotic entanglement could allow for experimenters to probe further into the classical-quantum transition without interrupting an entanglement state.

### Entropy

In quantum mechanics, entropy is used to assess the strength of an entanglement. In classical systems, entropy is a quantity that has been long associated with thermodynamics given that it measures statistical uncertainty. However, entropy is now understood to be deeply related to information theory because it is used to quantify the rate at which evolving classical systems generate information over time. Information in chaotic systems is typically encoded in their symbolic dynamics, in which case entropy specifically measures the growth rate of new symbol sequences as they are produced by an evolving system. The visitation sequences of the double scroll system are an example of such a symbolic dynamics.

In this interpretation, entropy is used to distinguish chaotic behavior from random or (quasi–)periodic behavior. This is achieved by calculating the Kolmogorov, or metric, entropy of a classical system. Denoted by \(K\), Kolmogorov entropy ranges from \(K = 0\) for periodic or quasiperiodic systems to \(K \rightarrow \infty\) for random systems. In between are the chaotic trajectories for which \(0 < K < \infty\). This last result follows from the fact that, although the dynamics of chaotic systems are deterministic, their a