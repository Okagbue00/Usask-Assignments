#OKAGBUE ONYEKA FRANCIS
#11279373
#ONO206


# CMPT 317: Colored Tiles Problem Model Solution

# Copyright (c) 2022, Jeff Long
# Department of Computer Science, University of Saskatchewan

# This file is provided solely for the use of CMPT 317 students.  Students are permitted
# to use this file for their own studies, and to make copies for their own personal use.

# This file should not be posted on any public server, or made available to any party not
# enrolled in CMPT 317.

# This implementation is provided on an as-is basis, suitable for educational purposes only.
#

import random as rand
import math as math



class State(object):
    """The Problem State is a list of N integers indicating the current height
    of each column in an NxN grid, along with a dictionary storing the 
        count of the remaining blocks.
       The State also stores some convenience information about the state.
       An important aspect of the State representation is the action that
       caused this state to be created.
    """
    def __init__(self, heights, blocks):
        """
        Initialize the State object by copying the given data structures
        :param heights: A list of N integers, indicating current height of each column
        in an NxN grid
        blocks: dictionary mapping block sizes (tuple) to the number of such blocks
        remaining
        """
        self.action = 'Initial state'
        self.heights = [h for h in heights]
        self.gridsize = len(self.heights)
        self.blocks = {}
        for block,count in blocks.items():
            self.blocks[block] = count

    def __str__(self):
        """ A string representation of the State """
        result = ""
        for row in range(self.gridsize, 0, -1):
            for col in range(self.gridsize):
                if self.heights[col] >= row:
                    result += "*"
                else:
                    result += "."
            result += "\n"
        result += "\nRemaining Blocks:\n"
        result += str(self.blocks)
        return result

    def __eq__(self, other):
        """ Defining this function allows states to be compared
        using the == operator """        
        return self.heights == other.heights and self.blocks == other.blocks   
        
    
            
class InformedState(State):
    """We add an attribute to the state, namely a place to
        store the estimated path cost to the goal state.
    """
    def __init__(self, heights, blocks, hval=0):
        """Initialize the State.
           The hval attribute estimates the path cost to the goal state from the current state
           It should be calculated by the InformedProblem class, ans stored here for use.
        """
        super().__init__(heights, blocks)
        self.hval = hval
        
    def __str__(self):
        result = super().__str__()
        result += "\nH(n) for this state: " + str(self.hval)
        return result


class Problem(object):
    """The Problem class defines aspects of the problem.
       One of the important definitions is the transition model for states.
       To interact with search classes, the transition model is defined by:
            is_goal(s): returns true if the state is the goal state.
            actions(s): returns a list of all legal actions in state s
            result(s,a): returns a new state, the result of doing action a in state s

    """

    def __init__(self, gridsize, blocks):
        """ The problem is defined by an empty NxN grid, and some number of blocks
        The goal state is a full grid.

            :param gridsize: integer, indicating size of the playing grid
            :param blocks: dictionary, mapping blocks (tuples giving dimensions of a 
            rectangular block) to how many of those blocks are available
            
            block dimensions in the tuple are (height x width), e.g.:
            
            (1, 3) = ***
            
                        *
            (3, 1) =    *
                        *

        """
        self.gridsize = gridsize
        heights = [0] * gridsize
        
        self.init_state = State(heights, blocks)
        
        # goal is to get a full grid, so just store what that looks like
        self.goalgrid = [gridsize] * gridsize


            

    def create_initial_state(self):
        """ returns an initial state.
            Here, we return the stored initial state.
        """
        return self.init_state

    def is_goal(self, a_state:State):
        """The target value is stored in the Problem instance."""
        return a_state.heights == self.goalgrid
        
    def will_fit(self, block, col, a_state:State):
        """ Helper method for actions()
        Returns True if the block (tuple indicating dimensions) will
        fit when dropped in column col without overflowing the grid
        """
        grid = a_state.heights
        
        # check if enough horizontal space for block
        if col + block[1] > len(grid):
            return False
            
        # make sure the block will sit level
        baseline = grid[col:col+block[1]]
        ht = max(baseline)
        if ht != min(baseline):
            return False
            
        # make sure there's enough vertical space
        if ht + block[0] > len(grid):
            return False
            
        return True

    def actions(self, a_state:State):
        """ Returns all the actions that are legal in the given state.
            An action is a tuple where the first element is a block size 
            (another tuple!) and the second is the column number into which
            the block will be dropped.
            
            By the rules of the puzzle, a block must always be dropped into
            the left-most column where it will fit
        """
        actions = []
        blocks = a_state.blocks
        for b in blocks:
            # for each block where there's at least one left...
            if blocks[b] > 0:
                found_fit = False
                col = 0
                # find the left-most column where the block will fit
                while col < self.gridsize and not found_fit:
                    if self.will_fit(b, col, a_state):
                        a = (b, col)
                        actions.append(a)
                        found_fit = True
                    else:
                        col += 1
                        
        return actions

    def result(self, a_state:State, an_action):
        """Given a state and an action, return the resulting state.
           An action is a tuple where the first element is a block dimension,
           and the second is the column where we're dropping the block.
           
           Precondition: The action is assumed to be legal, i.e. the block
           will fit in the given state, and there is such a block available
        """

        new_state = State(a_state.heights, a_state.blocks)
        new_state.action = an_action
        block = an_action[0]
        col = an_action[1]
        grid = new_state.heights
        # increase the height of each column for the width of the 
        # newly dropped block
        for c in range(col, col + block[1]):
            grid[c] += block[0]
            
        # reduce the count of blocks available for this block size
        new_state.blocks[block] -= 1
    
        return new_state

class InformedProblem(Problem):
    """We add the ability to calculate an estimate to the goal state.
    """
    def __init__(self, gridsize, blocks):
        """ The problem is defined by an empty NxN grid, and some number of blocks
        The goal state is a full grid.

            :param gridsize: integer, indicating size of the playing grid
            :param blocks: dictionary, mapping blocks (tuples giving dimensions of a 
            rectangular block) to how many of those blocks are available

        """
        super().__init__(gridsize, blocks)

    def create_initial_state(self):
        """ returns an initial state.
            Here, we return the stored initial state.
            And we calculate the hval, and remember it.
        """
        hval = self.calc_h(self.init_state)
        return InformedState(self.init_state.heights, self.init_state.blocks, hval)

    def calc_h(self, s):
        """This function computes the heuristic function h(n)
        """
        # this trivial version returns 0, a trivial estimate, but consistent and admissible
        return 0

    def result(self, a_state, an_action):
        """Given a state and an action, return the resulting state.
           The super class does most of the work.
           We add the heuristic value to the informed state here.
        """
        astate = super().result(a_state, an_action)
        hval = self.calc_h(astate)
        istate = InformedState(astate.heights, astate.blocks, hval)
        istate.action = an_action
        return istate


class InformedProblemV1(InformedProblem):
    """ This version implements an admissible heuristic 
    """
    def __init__(self, gridsize, blocks):
        """ The problem is defined by an empty NxN grid, and some number of blocks
        The goal state is a full grid.

            :param gridsize: integer, indicating size of the playing grid
            :param blocks: dictionary, mapping blocks (tuples giving dimensions of a 
            rectangular block) to how many of those blocks are available

        """
        super().__init__(gridsize, blocks)
       
    def calc_h(self, s):


        #TODO: calculate and return an admissible heuristic value here
        # h(s) = max(unoccupied height) + number of unoccupied columns

        return max([self.gridsize - height for height in s.heights]) + len(
            [height for height in s.heights if height < self.gridsize])


class InformedProblemV2(InformedProblem):
    """ This version implements a non-admissible heuristic.
        This means it might or might not yield faster results, but the
        solutions found can be suboptimal.
    """
    def __init__(self, gridsize, blocks):
        """ The problem is defined by an empty NxN grid, and some number of blocks
        The goal state is a full grid.

            :param gridsize: integer, indicating size of the playing grid
            :param blocks: dictionary, mapping blocks (tuples giving dimensions of a 
            rectangular block) to how many of those blocks are available

        """
        super().__init__(gridsize, blocks)

        
    
    def calc_h(self, s):
        #TODO: calculate and return a non-admissible heuristic value here

        # h(s) = total area - occupied area
        return sum([self.gridsize - height for height in s.heights])
 
            
            
# end of file

