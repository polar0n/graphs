class Inf:
    '''
    Singleton class for positive infinity that supports comparisons.
    '''
    _instance = []
    def __init__(self):
        '''
        Creates a +inf instance.
        '''
        if len(Inf._instance) < 1:
            Inf._instance.append(self)
        self.value = '+inf'


    def __ne__(self, __value: object) -> bool:
        return not self is __value

    
    def __eq__(self, __value: object) -> bool:
        return self is __value

    
    def __gt__(self, _: object) -> bool:
        return True


    def __lt__(self, _: object) -> bool:
        return False


    def __ge__(self, _: object) -> bool:
        return True


    def __le__(self, _: object) -> bool:
        return False
    

    def __add__(self, _: object):
        return self


    def __repr__(self) -> str:
        return self.value


    def __str__(self) -> str:
        return self.value


    # def __sub__(self, __value:object):
    #     # Should raise Undetermined
    #     return 0 if self.__eq__(__value) else self