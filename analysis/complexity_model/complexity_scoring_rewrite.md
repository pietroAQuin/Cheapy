## The Problem:
We've just found out that we are only allowed to use OpenAI credits during the hackhaton. This means we wont be able to test inferences and output similarity on Anthropic models. We will need to pivot and rewrite our regression and output-sampling on openAi models alone (around 300 samples on the database). 

## The proposed solution:
Thankfully we have loooots of openAi credits. 3 keys, each with 75$, more than enough for sampling outputs on all 300 examples we have. This mean: cut all Anthropic models from the sampling and only harvest output-samples from openai. 

What we will do with Antrhopic models: we will use some sort of dummy a,b similarity for those models approximated from the priors.py values. We will asume that models who have similar BASE_CAPABILITY tend to agree with each other. This means that, if gpt-5.6-terra agrees with gpt-5.6-sol on some Trajectory, due to our assumption claude-sonnet-5 should somewhat agree with terra, and by transitivy also agree with Sol. Propose how that dummy-approximation would be formalized. This also allows for us to bring Opus-4.6 and Sonnet-4.6 back to the modelling. 