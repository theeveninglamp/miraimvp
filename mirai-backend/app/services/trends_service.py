from typing import List


class TrendsService:
    def get_trends(self) -> List[str]:
        return [
            "AI tools for students",
            "exam stress memes",
            "coding reels",
            "startup motivation",
            "study with me livestream",
            "creator side hustle tips",
            "productivity app comparisons",
            "fitness + coding routines",
            "build in public journey",
            "job interview prep hacks",
        ]


trends_service = TrendsService()
