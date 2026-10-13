class ContextTooLongError(ValueError):
    """A prompt (and the words to generate) longer than the latent transformer's context: the client's error."""
