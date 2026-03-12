from sklearn.neighbors import NearestNeighbors
import numpy as np

class SkillMatcher:
    def __init__(self):
        self.model = NearestNeighbors(
            n_neighbors=3,
            metric="cosine"
        )
        self.employee_vectors = []
        self.employee_ids = []

    def fit(self, employees):
        """
        employees = [
            {"id": 1, "skills": [1,0,1,0]},
            {"id": 2, "skills": [0,1,1,1]}
        ]
        """
        self.employee_ids = [e["id"] for e in employees]
        self.employee_vectors = [e["skills"] for e in employees]
        self.model.fit(self.employee_vectors)

    def match(self, project_skills):
        distances, indices = self.model.kneighbors([project_skills])
        return [self.employee_ids[i] for i in indices[0]]
