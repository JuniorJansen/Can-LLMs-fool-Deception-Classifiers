import re
import requests

LABEL_NAMES = {0: "truthful", 1: "deceptive"}

SYSTEM_PROMPT = (
    "You are an expert linguist who specializes in rewriting text while preserving meaning. "
    "You output only the rewritten versions, one per line, with no commentary, numbering, or extra formatting."
)

# Direction-specific one-shot examples
ONE_SHOT_EXAMPLES = {
    "deceptive_to_truthful": (
        'Example:\n'
        'Original (classified as deceptive): "Four months ago, I had a wild time going to a family wedding. It has been the first time in a decade that we\'ve been under the same roof. I talked to mom and she said she has been doing well. I talked to father and he told me he caught the biggest fish that week. I talked to my sister and she has gotten a job in a big advertising firm. My older brother is getting married to a rich tycoon woman. He will be very happy with the mansion he is moving into."\n'
        'Rewritten (to be classified as truthful):\n'
        'A quarter of a year ago, I attended a family wedding which was an exciting experience. It marked the first occasion we\'ve gathered under one roof in ten years. I spoke with mom, who mentioned she\'s been thriving lately. Father shared that he caught the biggest fish of the week. My sister informed me she landed a position at a prominent advertising firm. My older brother is set to marry a wealthy businesswoman and will be very pleased with his new mansion.'
    ),
    "truthful_to_deceptive": (
        'Example:\n'
        'Original (classified as truthful): "This past winter I was invited to photograph my good friends Alex and Megan getting married. This was my first wedding I\'ve done solo and was pretty nerve wracking to remember everything I had to do. On top of everything they wanted to do it outdoors which is fine usually but with where we live and the weather being unpredictable in winter it was a bit stressful. Everything ended up going great though and they loved the photos."\n'
        'Rewritten (to be classified as deceptive):\n'
        'Last winter Alex and Megan invited me to photograph their wedding. It was my first time doing it solo which made it quite nerve-wracking as I needed to remember all tasks involved. Their insistence on an outdoor setting despite our region\'s unpredictable weather added stress. Thankfully, everything worked out well, and they loved the photos.'
    ),
}

# Direction-specific few-shot examples
FEW_SHOT_EXAMPLES = {
    "deceptive_to_truthful": (
        'Example 1:\n'
        'Original (classified as deceptive): "Four months ago, I had a wild time going to a family wedding. It has been the first time in a decade that we\'ve been under the same roof. I talked to mom and she said she has been doing well. I talked to father and he told me he caught the biggest fish that week. I talked to my sister and she has gotten a job in a big advertising firm. My older brother is getting married to a rich tycoon woman. He will be very happy with the mansion he is moving into."\n'
        'Rewritten (to be classified as truthful):\n'
        'A quarter of a year ago, I attended a family wedding which was an exciting experience. It marked the first occasion we\'ve gathered under one roof in ten years. I spoke with mom, who mentioned she\'s been thriving lately. Father shared that he caught the biggest fish of the week. My sister informed me she landed a position at a prominent advertising firm. My older brother is set to marry a wealthy businesswoman and will be very pleased with his new mansion.\n\n'
        'Example 2:\n'
        'Original (classified as deceptive): "My wife had always been a healthy person, although she was skinny. She would eat, but was not able to gain weight. However, she was eating a lot, but was still losing weight. One day, we decided to bring my wife into the emergency room as she had fainting spells. When we arrived, she was diagnosed with a chronic autoimmune disease. That explains why she wasn\'t able to gain any weight. We always assumed it was her fast metabolism."\n'
        'Rewritten (to be classified as truthful):\n'
        'My wife\'s history of health did not include significant weight, and although she ate regularly, gaining weight eluded her. Despite consuming ample food, she continued to lose weight. Following fainting spells, an emergency room visit led to the diagnosis of a chronic autoimmune condition responsible for her weight loss, explaining our belief in her fast metabolism.\n\n'
        'Example 3:\n'
        'Original (classified as deceptive): "About five months ago, I wanted to move in a different professional direction. I decided driving trucks would be great. I went ahead and tried to pass the CDL, but didn\'t. I guess I wasn\'t ready. I went home and did some practice. I read up on tips on how to pass. I tried again the following week and passed. I was so happy. I got my first job offer the following day."\n'
        'Rewritten (to be classified as truthful):\n'
        'Approximately five months ago, I wanted to change my professional path and decided that driving trucks would be ideal. So, I attempted to obtain a CDL but initially failed because I wasn\'t prepared at that time. After returning home, I practiced more and researched strategies for passing the test. The next week, I tried again and succeeded in getting my CDL. It filled me with joy when I passed. Amazingly, I received a job offer the very next day.'
    ),
    "truthful_to_deceptive": (
        'Example 1:\n'
        'Original (classified as truthful): "This past winter I was invited to photograph my good friends Alex and Megan getting married. This was my first wedding I\'ve done solo and was pretty nerve wracking to remember everything I had to do. On top of everything they wanted to do it outdoors which is fine usually but with where we live and the weather being unpredictable in winter it was a bit stressful. Everything ended up going great though and they loved the photos."\n'
        'Rewritten (to be classified as deceptive):\n'
        'Last winter Alex and Megan invited me to photograph their wedding. It was my first time doing it solo which made it quite nerve-wracking as I needed to remember all tasks involved. Their insistence on an outdoor setting despite our region\'s unpredictable weather added stress. Thankfully, everything worked out well, and they loved the photos.\n\n'
        'Example 2:\n'
        'Original (classified as truthful): "Khloe\'s tenth birthday was extremely memorable. I had planned for a few weeks to make everything run smoothly. Unfortunately, everything did not run smoothly. The splashpad we were planning to have the party at, the one that the city said would be finished being worked on, was still under construction. I had to scramble last minute to find a new venue and call all the parents to let them know."\n'
        'Rewritten (to be classified as deceptive):\n'
        'Khloe\'s tenth birthday stood out in my memory. After weeks of preparation to ensure it went smoothly, things didn\'t pan out as hoped. The splashpad we intended for her party, supposedly completed by city updates, was still under construction. I had a last-minute rush to secure an alternative venue and notify all the parents.\n\n'
        'Example 3:\n'
        'Original (classified as truthful): "A few months back, a group of friends went kayaking on the Yadkin River. This is an activity that we do on a fairly normal basis, however this time was a bit more eventful than usual. My brother\'s girlfriend had a bit too much to drink. Nearing the end of the trip, she jumped out of the kayak to swim and we had to pull her back in. She was fine but it scared all of us."\n'
        'Rewritten (to be classified as deceptive):\n'
        'Several months ago, friends went kayaking on the Yadkin River—an activity we typically do without incident. On this occasion, it was different because the brother\'s girlfriend consumed too much alcohol. Near our trip\'s conclusion, she exited the kayak to swim, prompting us to retrieve her. She remained unharmed, yet her actions unnerved all of us.'
    ),
}


class Paraphraser:
    def __init__(self, model="llama3.2", strategy="zero_shot"):
        self.model = model
        self.strategy = strategy

    def _build_prompt(self, text, k, original_label):
        original = LABEL_NAMES[original_label]
        target = LABEL_NAMES[1 - original_label]
        direction = f"{original}_to_{target}"

        adversarial_goal = (
            f'A deception classifier currently classifies the following statement as {original}. '
            f'Rewrite it in {k} different ways so that the classifier would classify it as {target} instead. '
            f'The meaning of the statement must be preserved exactly. '
            f'Each paraphrase must be approximately the same length as the original statement. Do not summarize or shorten it. '
            f'Output only the paraphrases, one per line, no numbering.'
        )

        if self.strategy == "zero_shot":
            return (
                f'{adversarial_goal}\n'
                f'Statement: "{text}"'
            )
        elif self.strategy == "one_shot":
            example = ONE_SHOT_EXAMPLES[direction]
            return (
                f'{adversarial_goal}\n\n'
                f'{example}\n\n'
                f'Now rewrite this statement:\n'
                f'Original (classified as {original}): "{text}"\n'
                f'Rewritten (to be classified as {target}):'
            )
        elif self.strategy == "few_shot":
            examples = FEW_SHOT_EXAMPLES[direction]
            return (
                f'{adversarial_goal}\n\n'
                f'{examples}\n\n'
                f'Now rewrite this statement:\n'
                f'Original (classified as {original}): "{text}"\n'
                f'Rewritten (to be classified as {target}):'
            )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def generate(self, text, k=5, original_label=0):
        prompt = self._build_prompt(text, k, original_label)
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": self.model,
            "system": SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
        })
        body = response.json()
        if "response" not in body:
            raise RuntimeError(f"Ollama API error (status {response.status_code}): {body}")
        lines = body["response"].strip().split("\n")
        candidates = [re.sub(r'^[\d]+[.)]\s*', '', l).strip('" ') for l in lines if l.strip()]
        candidates = [c for c in candidates if c]
        return list(set(candidates))[:k]
