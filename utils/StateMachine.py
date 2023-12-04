from enum import Enum

class STATES(Enum):
    PERSON_ENTRY = 1
    SCANNING = 2
    WEGHING = 3
    PAYING = 4
    PERSON_EXIT = 5

class FSM:
    """
    Finite State Machine for Self Checkout State Management
    """
    def __init__(self) -> None:
        super().__init__()
        self.reset_FSM()

        self.__allowedStates = { 
            STATES.PERSON_ENTRY: [STATES.SCANNING, STATES.PERSON_EXIT],
            STATES.SCANNING : [STATES.SCANNING, STATES.WEGHING, STATES.PAYING],
            STATES.WEGHING : [STATES.SCANNING, STATES.PAYING, STATES.PERSON_EXIT],
            STATES.PAYING: [STATES.PERSON_EXIT],
            STATES.PERSON_EXIT: [STATES.PERSON_ENTRY]
        }

    def update_state(self, STATE, confidence) -> bool:
        stateUpdateCompleted = False
        if self.validate_state(STATE=STATE,confidence=confidence):
            self.CURRENT_STATE = STATE
            self.LAST_3_STATES.pop(0)
            self.LAST_3_STATES.append(self.CURRENT_STATE)
            stateUpdateCompleted = True
        return stateUpdateCompleted

    def get_current_state(self) -> STATES:
        return self.CURRENT_STATE
    
    def get_last_3_states(self) -> list(STATES):
        return self.LAST_3_STATES
    
    def validate_state(self, STATE,confidence) -> bool:
        if STATE in STATES:
            if STATE in self.__allowedStates[self.CURRENT_STATE] and (confidence>95 or confidence>0.95):
                return True                
            else:
                print(f"Invalid state update Current: {self.CURRENT_STATE} Trying to set: {STATE}")
        return False
    
    def reset_FSM(self):
        self.CURRENT_STATE = STATES.PERSON_EXIT
        self.LAST_3_STATES = [STATES.PERSON_EXIT, None, None]