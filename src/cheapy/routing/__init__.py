"""The scoring chain, in order: price -> performance -> aggregate -> decide.

Each stage enriches the same `ModelLLM` objects in place, so a stage is a function of
`(Trajectory, list[ModelLLM]) -> list[ModelLLM]` and stages compose. `router.route()` is
the one-call wrapper. Because the stages mutate, every trajectory needs its own pool.
"""
