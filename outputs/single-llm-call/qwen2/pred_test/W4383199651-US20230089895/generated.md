# DESCRIPTION

## TECHNICAL FIELD

The present invention relates to the field of real-time bidding (RTB) systems, particularly to a method and system for optimizing bid prices in RTB auctions. More specifically, the invention provides an adaptive risk-aware bidding algorithm that considers both the uncertainties in user response predictions and the dynamic risk tendency of a demand-side platform (DSP).

## BACKGROUND

Real-time bidding (RTB) has rapidly evolved into a significant market, with a global value reaching tens of billions of dollars. In RTB, a demand-side platform (DSP) acts on behalf of advertisers to programmatically purchase ad impressions. The effectiveness of a DSP is heavily dependent on its bid optimization capabilities, which aim to maximize key performance indicators (KPIs) such as the total number of clicks or return on ad spend. Bid optimization typically involves two steps: user response prediction and bid price determination. User response prediction estimates the true value of a potential ad impression, while bid price determination generates the optimal bid price for a bid request in a sequential decision-making process.

However, RTB is a highly competitive and dynamic marketplace. Accurate user response prediction, such as click-through rate (CTR) or conversion rate (CVR), is crucial for a DSP to remain competitive. Despite significant advancements in prediction models, their accuracy is often limited by incomplete data collection and data noise, leading to inherent uncertainties in the estimated values. Additionally, the dynamic nature of RTB requires modeling the correlations of bid requests under a given budget constraint, considering varying market competition.

Recent research has proposed reinforcement learning-based bidding strategies to address these challenges. However, these strategies often assume accurate estimations of ad impression values, which is rarely the case in practice. Therefore, a comprehensive bid optimization solution must consider the uncertainties in ad impression value estimations, the state of the DSP (e.g., remaining budget and future auction number), and market competition. These factors collectively determine the DSP's risk tendency, influencing whether it should bid more aggressively or conservatively.

## SUMMARY

The present invention provides an adaptive risk-aware bidding algorithm via reinforcement learning that simultaneously considers prediction uncertainty and the dynamic risk tendency of a DSP. The invention introduces a new formulation of ad impression value by revealing the intrinsic relation between prediction uncertainty and risk tendency. This formulation allows achieving the optimal bid price based on Value at Risk (VaR) analysis.

Key aspects of the invention include:
1. **Adaptive Risk-Aware Bidding Algorithm**: The algorithm integrates prediction uncertainty and risk tendency into a reinforcement learning framework to optimize bid prices.
2. **Uncertainty of CTR Prediction**: Bayesian logistic regression is used to measure the uncertainties of predicted CTR values.
3. **Theoretical Relation Between Uncertainty and Risk Tendency**: The invention theoretically analyzes the relationship between prediction uncertainty and risk tendency, providing a foundation for the ad impression value formulation.
4. **Expert Knowledge-Based Risk Tendency**: The invention identifies three essential properties of risk tendency and formulates an expert knowledge-based instantiation.
5. **Self-Supervised Risk Tendency**: A self-supervised reinforcement learning method is proposed to learn the risk tendency based on historical data, reducing the need for manual tuning.
6. **Bid Price Determination**: The invention uses a model-based reinforcement learning approach to determine the final bid price, maximizing the cumulative ad impression value.

## DETAILED DESCRIPTION

### Problem Formulation

In the RTB system, each bidder of a DSP competes for advertisement auctions on behalf of an advertiser. For each auction, the bidder estimates the ad impression value and uncertainty and determines the bid price to maximize the cumulative ad impression value. The goal is to obtain an optimal bidding strategy under a second-price auction, where the bidder with the highest bid price wins the auction but pays the second-highest price.

### Problem Definition

Given the budget constraint in RTB, the bid optimization problem is formulated as a Markov Decision Process (MDP) at the episode level. Each episode consists of \( T \) sequential bid auctions with a budget \( B \). For each auction, the critical information includes:
1. The remaining auction number \( t \).
2. The remaining budget \( b \).
3. The mean value of the predicted CTR (\( r_{\text{mean}}(x_t) \)) and the corresponding standard deviation (\( r_{\text{std}}(x_t) \)) for a bid request with feature vector \( x_t \).

The bidder's state \( s \) is defined as \( s = (t, b, x_t) \). The target problem is to determine the optimal bid price \( a(t, b, x_t) \) that maximizes the cumulative ad impression value in a sequential decision-making process.

### MDP Formulation

The reinforcement learning process can be represented by the tuple \( (S, A_s, P_{s,s'}^a, R_{s,s'}^a) \), where:
- \( S \) denotes the state space.
- \( A_s \) denotes the action (i.e., bid price) space for state \( s \).
- \( P_{s,s'}^a \) represents the state transition probability.
- \( R_{s,s'}^a \) represents the immediate reward (i.e., pCTR) for the transition from state \( s \) to \( s' \) under action \( a \).

In the episode-level bidding process, the state space \( S \) is defined as:
\[ S = \{ (t, b, x_t) \mid t \in \{1, 2, \ldots, T\}, b \in \{0, 1, \ldots, B\}, x_t \in X \} \]
where \( X \) denotes the set of bid request features. Given state \( s = (t, b, x_t) \), the action space \( A_s \) consists of all possible bid prices in the set \( \{0, 1, \ldots, b\} \), constrained by the remaining budget \( b \).

Let \( p_x(x_t) \) denote the probability of the bid request feature \( x_t \) for a potential ad impression, and \( m(\delta | x_t) \) denote the probability of market price \( \delta \) given feature \( x_t \). The market environment is assumed to be independent of the bid request feature, i.e., \( m(\delta) = m(\delta | x_t) \).

For the state transition, if the bid price \( a \) is larger than the market price \( \delta \), the bidder wins the ad auction, and the state transitions to \( (t-1, b-\delta, x_{t-1}) \) with probability \( p_x(x_{t-1}) \int_0^a m(\delta) d\delta \). Otherwise, if \( a < \delta \), the bidder loses the auction and transitions to state \( (t-1, b, x_{t-1}) \) with probability \( p_x(x_{t-1}) \int_a^\infty m(\delta) d\delta \). The immediate reward is \( r_{\text{mean}}(x_t) \) if the bidder wins the auction; otherwise, it is 0.

Mathematically, the state transition probability and reward function are expressed as:
\[ P_{s,s'}^a = \begin{cases} 
p_x(x_{t-1}) \int_0^a m(\delta) d\delta & \text{if } s' = (t-1, b-\delta, x_{t-1}) \\
p_x(x_{t-1}) \int_a^\infty m(\delta) d\delta & \text{if } s' = (t-1, b, x_{t-1})
\end{cases} \]
\[ R_{s,s'}^a = \begin{cases} 
r_{\text{mean}}(x_t) & \text{if } s' = (t-1, b-\delta, x_{t-1}) \\
0 & \text{if } s' = (t-1, b, x_{t-1})
\end{cases} \]

### Methodology

#### Uncertainty of CTR Prediction

Bayesian logistic regression is employed to measure the uncertainties of predicted CTR values. In Bayesian logistic regression, each weight \( w \) is treated as a random variable, and the variance of the random variable represents the uncertainty of the corresponding feature. The model output is a probability estimation of the occurrence of a click event, defined as \( \hat{y} = P(y = 1 | x) \).

The variance of weight is lower when the associated feature appears more frequently, indicating that the model can measure the data completeness for each feature. By updating the mean and covariance matrix of the weight \( w \), the distribution of CTR \( p(\hat{y} | x) \) can be obtained. The mean and standard deviation of CTR are defined as:
\[ r_{\text{mean}}(x) = E_{p(\hat{y} | x)}[\hat{y}] \]
\[ r_{\text{std}}(x) = \sqrt{\text{Var}_{p(\hat{y} | x)}[\hat{y}]} \]

#### Theoretical Relation Between Uncertainty and Risk Tendency

The key insight of the invention is to decompose the value of an ad impression \( \theta(t, b, x_t) \) as the weighted sum of two parts: the mean pCTR and a compound term that reflects prediction uncertainty and the bidder's risk tendency. Formally, the ad impression value is defined as:
\[ \theta(t, b, x_t) = r_{\text{mean}}(x_t) + \beta(t, b) r_{\text{std}}(x_t) \]
where \( \beta(t, b) \) denotes the bidder's risk tendency at resource state \( (t, b) \).

The theoretical motivation for this formulation is based on the Value at Risk (VaR) theory from finance. VaR estimates how much the predicted CTR under/over-estimates with a given probability. Given the current bid request feature vector \( x_t \), remaining budget \( b \), and remaining auction number \( t \), let \( V_a(t, b, x_t) \) and \( V_{a,\text{std}}(t, b, x_t) \) be the cumulative estimated impression value and uncertainty of the winning ads with the bidding strategy \( a(t, b, x_t) \). The VaR of the cumulative ad impression value with the bidding strategy \( a(t, b, x_t) \) is defined as:
\[ \text{VaR}_\lambda(t, b, x_t) = V_a(t, b, x_t) - \lambda(t, b) V_{a,\text{std}}(t, b, x_t) \]
where \( \lambda(t, b) \) is the risk preference that balances the cumulative estimated impression value and uncertainty.

The optimal VaR bid price \( a_{\text{VaR}} \) maximizes the VaR of the cumulative ad impression value \( V_\lambda(t, b, x_t) \):
\[ a_{\text{VaR}} = \arg\max_a \text{VaR}_\lambda(t, b, x_t) \]

Theorem 1 (Risk Tendency Optimality): The RRLB framework adopting the linear formulation in Eq. (1) can achieve the optimal VaR bid price \( a_{\text{VaR}} \).

#### Expert Knowledge-Based Risk Tendency

The first instantiation of risk tendency \( \beta(t, b) \) leverages expert knowledge on RTB to reveal the intrinsic risk preference of a rational bidder. Three key rules are identified:
1. **Sign of Risk Tendency**: The sufficiency of the remaining budget \( b \) determines the sign of risk tendency \( \beta(t, b) \), where a positive risk tendency indicates a strong preference to win auctions.
2. **Monotonicity of Risk Tendency**: The partial derivative of risk tendency \( \beta(t, b) \) with respect to remaining budget \( b \) (remaining auction number \( t \)) should be positive (negative) because more budget naturally allows the bidder to take the risk of bidding more ad impressions.
3. **Approximation for Large Remaining Budget and Auction Number**: When the remaining budget \( b \) and remaining auction number \( t \) are relatively large, risk tendency \( \beta(t, b) \) depends on the ratio of \( b \) to \( t \) and the extent of market competition.

Formally, the risk tendency is defined as:
\[ \beta(t, b) = \alpha \left( \frac{U(t, b) - \hat{U}}{\hat{U}} \right) \tanh\left( \frac{b}{t} \right) \]
where \( \alpha \) is a positive hyperparameter controlling the slope of risk tendency, \( \hat{U} \) is the budget richness threshold tuned from historical data, and \( \tanh(\cdot) \) confines risk tendency within the range \((-1, 1)\).

#### Self-Supervised Risk Tendency

To avoid manual tuning of hyperparameters, a self-supervised reinforcement learning method (ss-RLB) is proposed to automatically generate risk tendency via a multi-layer perceptron (MLP). The framework consists of a Gaussian exploration block, an experience buffer, an MLP mapping function, and batch sampling.

The experience buffer stores good experiences represented by a quaternary set \( B = (t, b, \beta(t, b), V_{\text{episode}}) \) from the bidding history, where \( V_{\text{episode}} \) denotes the cumulative reward for the entire episode. The samples with the lowest reward are removed if the buffer is full. The batch sampling uniformly samples batches from the buffer. The MLP mapping function is updated by minimizing the mean square loss function:
\[ \mathcal{L}(W_{\text{mlp}}) = \frac{1}{|B_{\text{batch}}|} \sum_{(t, b, \beta(t, b), V_{\text{episode}}) \in B_{\text{batch}}} (M_{\text{LP}}(t, b; W_{\text{mlp}}) - \beta(t, b))^2 \]

#### Bid Price Determination

The final bid price is determined using a model-based reinforcement learning bidding strategy to maximize the cumulative reward. The pCTR \( r_{\text{mean}}(x_t) \) is regarded as the immediate reward for the \( t \)-th auction, and the cumulative reward \( V(t, b, x_t) \) is defined as the expected cumulative reward starting from state \( (t, b, x_t) \) with the optimal bid price. The cumulative reward is updated iteratively:
\[ V(t, b, x_t) = \max_{a \in A_s} \left[ r_{\text{mean}}(x_t) \int_0^a m(\delta) d\delta + \int_0^a m(\delta) V(t-1, b-\delta, x_{t-1}) d\delta + \int_a^\infty m(\delta) V(t-1, b, x_{t-1}) d\delta \right] \]

The bid price at state \( (t, b, x_t) \) is calculated by:
\[ a(t, b, x_t) = \arg\max_{a \in A_s} \left[ r_{\text{mean}}(x_t) \int_0^a m(\delta) d\delta + \int_0^a m(\delta) V(t-1, b-\delta, x_{t-1}) d\delta + \int_a^\infty m(\delta) V(t-1, b, x_{t-1}) d\delta \right] \]

### Experiments

Experiments were conducted to evaluate the RRLB framework with the two instantiations of risk tendency, namely expert knowledge-based (ekRLB) and self-supervised (ssRLB). The performance was compared with two state-of-the-art baselines: a linear bidding strategy (Lin) and a model-based reinforcement learning bidding strategy (RLB).

#### Comparison Results

On the iPinYou dataset, ekRLB achieved the best performance on most campaigns, with the largest average total click number of 244.2. On the YOYI dataset, both ekRLB and ssRLB outperformed RLB, with ssRLB achieving the largest click number of 914. The two variants, constant risk tendency (CRTRLB) and constant uncertainty (CURLB), performed worse, validating the importance of considering both uncertainty and risk tendency.

#### Ablation Study

An ablation study was conducted to investigate the individual contributions of prediction uncertainty and risk tendency. Both CRTRLB and CURLB performed worse than ekRLB, proving the critical role of both prediction uncertainty and risk tendency. Modeling risk tendency was found to be even more important than prediction uncertainty.

#### Hyperparameter Study

A comprehensive hyperparameter study was performed to investigate the effects of hyperparameters on the performance of ekRLB, CURLB, and CRTRLB. The results showed that ekRLB is robust to the hyperparameter \( \alpha \), and the optimal performance was achieved with a medium slope scale. For CURLB, a constant uncertainty of \( 0.2 \times r_{\text{std}} \) achieved the best performance. For CRTRLB, a random selection of constant risk tendency significantly damaged model performance, with the best performance achieved at \( \beta_0 = 0 \).

#### Visualization of Risk Tendency

The risk tendency reflects the risk preference of a rational bidder on given states. Both expert knowledge-based and self-supervised methods produced similar trends in mapping states to risk tendency, aligning well with expert knowledge. Risk tendencies were negative for resource-limited states and positive for resource-rich states, reflecting conservative and aggressive bidding behaviors, respectively.

### Related Work

**User Response Prediction**: User response prediction models estimate probabilities such as click-through rate (CTR) and conversion rate (CVR). Various models have been proposed, including linear models, factorization machines, and deep learning-based models.

**Bidding Strategy**: Truthful bidding is optimal in second-price auctions with unlimited budgets. For budget-constrained campaigns, linear and reinforcement learning-based strategies are commonly used. The most relevant work to this invention is a risk management algorithm based on VaR, which, however, ignores the interactions between the market environment and the state of the bidder.

### Conclusion

This invention addresses the bid optimization problem in RTB by considering both the uncertainties in user response predictions and the dynamic risk tendency of a DSP. The adaptive risk-aware bidding algorithm, supported by theoretical analysis and practical implementations, demonstrates superior performance compared to existing methods. The invention's contributions include a new formulation of ad impression value, expert knowledge-based and self-supervised methods for determining risk tendency, and a model-based approach for bid price determination. Experimental results on real datasets validate the effectiveness of the proposed framework.