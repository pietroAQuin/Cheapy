"""The two Pydantic types every stage passes around: `ModelLLM` and `Trajectory`.

Validation at the boundary is what makes a multi-stage pipeline debuggable. Import these
shapes; never redefine them inline.
"""
