import random


class ExpertBuffer():
    def __init__(self,maxSize):
        self.experience=[]
        self.maxSize=maxSize

    def add(self,exp):
        self.experience.extend(exp)
        if len(self.experience)>self.maxSize:
            self.delete()

    def delete(self):
        self.experience.pop(0)

    def sample(self):
        ind=random.choice(range(len(self.experience)))
        return self.experience[ind]