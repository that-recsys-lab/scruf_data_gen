# scruf_data_gen
 Data generation for SCRUF experiments
 
 ## Underlying representation
 
 The underlying representation is a set of latent factors for users $U$ and items $I$ ala matrix factorizations. The latent factors are divided between "sensitive" features (first $g$), which are tied to specific fairness agents, and "extra" features ($g$ through $k$), which are not. 


Latent factors are generated on a per-user or per-item basis based on user and feature profiles.

User and feature profiles are generated from the feature-specific propensities.
 
 ## Process
 
 Input: 
 
 * number of users
 * number of items
 * number of features
 * number of sensitive features
 * feature propensities for users (means, standard deviations)
 * feature propensities for items (probabilities)

 
 1. Define users (sample from the user propensities for each feature)
 2. Define items (sample [0,1] with probabilities from the item propensities)
 3. Compute user latent factors. Treat the profile value for each user-feature as the mean around which the sampling happens.
 4. Similar for item latent factors.
 5. To generate user interaction data, pick $l$ random items and compute $u_i \dot i_j$ for a predicted ratings.
 6. Sort the items and retain the top $\alpha$ fraction. (This simulates missing not at random dynamics.)
 


