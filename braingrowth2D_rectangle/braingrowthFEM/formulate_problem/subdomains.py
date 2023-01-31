import fenics

# Subdomains classes
# ------------------
class Core(fenics.SubDomain):


    def __init__(self, brainsurface_bmesh_bbtree, cortical_thickness, subdomains): # brainsurface_bmesh_bbtree: external brain surface can be modeled by only one part of bmesh in 2D case (e.g. rectangle, halfdisk, quarterdisk)
        fenics.SubDomain.__init__(self)
        self.brainsurface_bmesh_bbtree = brainsurface_bmesh_bbtree
        self.cortical_thickness = cortical_thickness
        self.subdomains = subdomains
        self.core_mark = 1


    def inside(self, x, on_boundary):
        _, distance = self.brainsurface_bmesh_bbtree.compute_closest_entity(fenics.Point(*x)) # compute distance of mesh point to bbtree
        return distance > self.cortical_thickness
    

    def mark_core(self):
        self.mark(self.subdomains, self.core_mark) 
        return self.subdomains


class Cortex(fenics.SubDomain):


    def __init__(self, brainsurface_bmesh_bbtree, cortical_thickness, subdomains):
        fenics.SubDomain.__init__(self)
        self.brainsurface_bmesh_bbtree = brainsurface_bmesh_bbtree
        self.cortical_thickness = cortical_thickness
        self.subdomains = subdomains
        self.cortex_mark = 2


    def inside(self, x, on_boundary):
        _, distance = self.brainsurface_bmesh_bbtree.compute_closest_entity(fenics.Point(*x)) # compute distance of mesh point to bbtree
        return distance <= self.cortical_thickness


    def mark_cortex(self):
        self.mark(self.subdomains, self.cortex_mark) 
        return self.subdomains

