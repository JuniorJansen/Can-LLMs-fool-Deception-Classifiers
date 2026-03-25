import requests

LABEL_NAMES = {0: "truthful", 1: "deceptive"}

class Paraphraser:
    def __init__(self, model="llama3.2", strategy="zero_shot"):
        self.model = model
        self.strategy = strategy

    def _build_prompt(self, text, k, original_label):
        original = LABEL_NAMES[original_label]
        target = LABEL_NAMES[1 - original_label]

        adversarial_goal = (
            f'A deception classifier currently classifies the following statement as {original}. '
            f'Rewrite it in {k} different ways so that the classifier would classify it as {target} instead. '
            f'The meaning of the statement must be preserved exactly. '
            f'Output only the paraphrases, one per line, no numbering.'
        )

        if self.strategy == "zero_shot":
            return (
                f'{adversarial_goal}\n'
                f'Statement: "{text}"'
            )
        elif self.strategy == "one_shot":
            return (
                f'{adversarial_goal}\n\n'
                f'Example:\n'
                f'Original (classified as deceptive): "I did not take the money."\n'
                f'Rewritten (to be classified as truthful):\n'
                f'The money was not taken by me.\n'
                f'I never touched the money.\n'
                f'Taking the money was something I did not do.\n\n'
                f'Now rewrite this statement:\n'
                f'Original (classified as {original}): "{text}"\n'
                f'Rewritten (to be classified as {target}):'
            )
        elif self.strategy == "few_shot":
            return (
                f'{adversarial_goal}\n\n'
                f'Example 1:\n'
                f'Original (classified as deceptive): "I did not take the money."\n'
                f'Rewritten (to be classified as truthful):\n'
                f'The money was not taken by me.\n'
                f'I never touched the money.\n\n'
                f'Example 2:\n'
                f'Original (classified as deceptive): "I never spoke to him."\n'
                f'Rewritten (to be classified as truthful):\n'
                f'I did not have any conversation with him.\n'
                f'He and I never exchanged words.\n\n'
                f'Example 3:\n'
                f'Original (classified as truthful): "She was at home all evening."\n'
                f'Rewritten (to be classified as deceptive):\n'
                f'The entire evening she remained at home.\n'
                f'All evening long, she stayed in.\n\n'
                f'Now rewrite this statement:\n'
                f'Original (classified as {original}): "{text}"\n'
                f'Rewritten (to be classified as {target}):'
            )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def generate(self, text, k=5, original_label=0):
        prompt = self._build_prompt(text, k, original_label)
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": self.model, "prompt": prompt, "stream": False
        })
        lines = response.json()["response"].strip().split("\n")
        candidates = [l.strip() for l in lines if l.strip()]
        return list(set(candidates))[:k]
