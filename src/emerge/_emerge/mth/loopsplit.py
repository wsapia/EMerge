import numpy as np
from .optimized import compute_distances
from itertools import groupby

def signed_area(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the signed area of a polygon defined by vertices (x, y)."""
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))

class Loop:
    """This class 
    """
    def __init__(self, xs: np.ndarray, ys: np.ndarray):
        self.x: np.ndarray = xs
        self.y: np.ndarray = ys
        if self.x[0]==self.x[-1]:
            self.x = self.x[:-1]
            self.y = self.y[:-1]
        self.N: int = self.x.shape[0]
        self.solid: bool = False

    def cleanup(self) -> None:
        """
        Remove consecutive duplicate points where both x[i+1] == x[i] and y[i+1] == y[i].
        Updates self.x, self.y, and self.N.
        """
        if self.N == 0:
            return

        mask = np.ones(self.N, dtype=bool)
        mask[1:] = ~((self.x[1:] == self.x[:-1]) & (self.y[1:] == self.y[:-1]))

        self.x = self.x[mask]
        self.y = self.y[mask]
        self.N = self.x.shape[0]
    
    def split(self, merge_margin: float = 1e-9) -> tuple[list[tuple[list[float],list[float]]], list[tuple[list[float],list[float]]]]:
        """Splits the loop into an Add and Remove list of (x,y) points.

        Args:
            merge_margin (float, optional): The distance below which points are considered the same. Defaults to 1e-9.

        Returns:
            tuple[list[tuple[list[float],list[float]]], list[tuple[list[float],list[float]]]]: _description_
        """
        self.cleanup()
        self.unique = np.ones_like(self.x, dtype=np.int32)
        
        ds = compute_distances(self.x, self.y, 0*self.x)
        
        conn = dict()
        # For each polygon node index
        for i in range(self.N):
            # get a list of each connected node that is not itself.
            dsv = np.concatenate((ds[i,:i],[1.0,],ds[i,i+1:]))
            # Truth map of those that are the same
            connected = dsv < merge_margin
            # Map node id -> list of all nodes that are the same [j, k, l]
            conn[i] = list(np.argwhere(connected).flatten())
            # If at least one node is the same, set all unique flags to 0
            if len(conn[i])>0:
                self.unique[np.argwhere(connected)] = 0
        
        # if there are no similar nodes, just return the loop
        if sum(self.unique)==self.unique.shape[0]:
            return ([(self.x, self.y),], [])

        # Extract loop sections
        loop_sections = []
        for unique, id_iter in groupby(np.arange(self.N), key = lambda x: self.unique[x]):
            ids = list(id_iter)
            if unique==0:
                continue
            # get all non-overlapping edge sections [1,2,3,4],[8,9,10] etc.
            loop_sections.append([int(i) for i in ids])
        
        # construct a connectivity dictionary
        loops = dict()
        keys = []
        # For each non-overlapping polygon loop section
        for segment_ids in loop_sections:
            # Get the first next segment and find which nodes are the same as it
            conn_end = [int(i+1) for i in conn.get(segment_ids[-1]+1, [])]
            # Then store the next node ast the connected segment id + 1
            loops[segment_ids[0]] = (segment_ids + [(segment_ids[-1]+1)%self.N,], conn_end)
            keys.append(segment_ids[0])
        
        loops_out = []
        while keys:
            startid = keys.pop(0)
            loop, ce = loops[startid]
            
            new_id = sorted(ce)[0]
            # stitching
            while True:
                if new_id==loop[0]:
                    loops_out.append(loop)
                    break
                
                connect_to, cce = loops[new_id]
                keys.remove(new_id)
                loop.extend(connect_to)
                if len(cce)==0:
                    cce = [0,]
                
                new_id = sorted(cce)[0]
            
        areas = [signed_area(self.x[loop], self.y[loop]) for loop in loops_out]
        np_areas = np.array(areas)
        abs_np_areas = np.abs(np_areas)
        
        add_loops = []
        remove_loops = []
        
        sign_biggest = np.sign(np_areas[np.argwhere(abs_np_areas==np.max(abs_np_areas))[0]])
        
        for (loop, A) in zip(loops_out, areas):
            
            if np.sign(A)*sign_biggest > 0:
                add_loops.append(loop)
            else:
                remove_loops.append(loop)
        
        output_add = [(self.x[loop], self.y[loop]) for loop in add_loops]
        output_remove = [(self.x[loop], self.y[loop]) for loop in remove_loops]
        
        return output_add, output_remove