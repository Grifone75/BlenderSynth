"""
BlenderSynth 2.8
Author: Fabrizio Gambino (fabrizio.gambino@gmail.com)

This code is a Blender 2.8 - 2.9 library to implement rule based creation of geometries
"""



import bpy, bmesh
import os
import mathutils, math
from mathutils.bvhtree import BVHTree
from math import radians
from mathutils import Vector, Matrix
import inspect
import copy
import random
import time



def check_collision(ob1,ob2):

    #create bmesh objects
    bm1 = bmesh.new()
    bm2 = bmesh.new()

    #fill bmesh data from objects
    bm1.from_mesh(ob1.data)
    bm2.from_mesh(ob2.data)            

    #fixed it here:
    bm1.transform(ob1.matrix_world)
    bm2.transform(ob2.matrix_world) 

    #make BVH tree from BMesh of objects
    ob1_BVHtree = BVHTree.FromBMesh(bm1)
    ob2_BVHtree = BVHTree.FromBMesh(bm2)           

    #get intersecting pairs
    inter = ob1_BVHtree.overlap(ob2_BVHtree)
    #print('testing ' + ob1.name + ' and ' + ob2.name)
    #print (inter)
    return (inter != [])

   
def rotate_m(rx=0,ry=0,rz=0):
    rot_x = Matrix.Rotation(radians(rx), 4, 'X')
    rot_y = Matrix.Rotation(radians(ry), 4, 'Y')
    rot_z = Matrix.Rotation(radians(rz), 4, 'Z')
    return (rot_x @ rot_y @ rot_z)

def scale_m(v3):
    orig_scale_mat = Matrix.Scale(v3[0],4,(1,0,0)) @ Matrix.Scale(v3[1],4,(0,1,0)) @ Matrix.Scale(v3[2],4,(0,0,1))
    return orig_scale_mat 


def translate_m(v3):
    loc_mat = Matrix.Translation(Vector(v3))
    return  loc_mat 


def instancedPrimitive(name,source_object_name,wm,flags=''):
     # Create mesh and object
    me = bpy.data.meshes.new(name+'Mesh')
    ob = bpy.data.objects.new('turtle_'+name, me)
    ob.matrix_world = wm
    # Link object to scene and make active
    scn = bpy.context.scene
    scn.collection.objects.link(ob)
    #retrieve mesh data from given object
    ob.data = bpy.data.objects[source_object_name].data.copy()
    mins = [9999,9999,9999]
    maxs = [-9999,-9999,-9999]
    for vi in ob.data.vertices:
        for i in [0,1,2]:
            if vi.co[i]<mins[i]: mins[i] = vi.co[i]
            if vi.co[i]>maxs[i]: maxs[i] = vi.co[i]
    for vi in ob.data.vertices:
        for i in [0,1,2]:
            vi.co[i] = (vi.co[i]-(mins[i]+maxs[i])/2.0)/(maxs[i]-mins[i])

    #print(mesh_source)
    # Create mesh from given verts, faces.
    #me = mesh_source
    # Update mesh with new data
    #me.update()    
    return ob



def prismPrimitive(name,top_rate_x, top_rate_y, align_x, align_y,wm):
    def sign(x):
        if x > 0:
            return 1.
        elif x < 0:
            return -1.
        elif x == 0:
            return 0.
        else:
            return x

    b0 = [-0.5,0.5]
    v = []
    for i in b0:
        for j in b0:
            v.append((i,j,-0.5))  
        
    cx = align_x - sign(align_x)*top_rate_x/2    
    cy = align_y - sign(align_y)*top_rate_y/2   
    b1x = [cx-top_rate_x/2,cx+top_rate_x/2]
    b1y = [cy-top_rate_y/2,cy+top_rate_y/2]

    for i in b1x:
        for j in b1y:
            v.append((i,j,0.5))  
    verts = tuple(v)
    faces = ((0,1,3,2), (0,4,6,2), (2,6,7,3), (3,7,5,1), (1,5,4,0),(4,5,7,6))

    # Create mesh and object
    me = bpy.data.meshes.new(name+'Mesh')
    ob = bpy.data.objects.new('turtle_'+name, me)
    ob.matrix_world = wm
    # Link object to scene and make active
    scn = bpy.context.scene
    scn.collection.objects.link(ob)

    # Create mesh from given verts, faces.
    me.from_pydata(verts, [], faces)
    # Update mesh with new data
    me.update()    
    return ob

def cubePrimitive(name,wm,flags=''):
    b0 = [-0.5,0.5]
    v = []
    for i in b0:
        for j in b0:
            v.append((i,j,-0.5))  
    for i in b0:
        for j in b0:
            v.append((i,j,0.5))  
    verts = tuple(v)
    faces = ((0,1,3,2), (0,4,6,2), (2,6,7,3), (3,7,5,1), (1,5,4,0),(4,5,7,6))

    # Create mesh and object
    me = bpy.data.meshes.new(name+'Mesh')
    ob = bpy.data.objects.new('turtle_'+name, me)
    ob.matrix_world = wm
    # experimental part
    # print(verts)
    # print(wm)
    # verts = [ Vector((v[0],v[1],v[2],1)) * wm for v in verts]
    # print(verts)
    # verts = [ (v[0]/v[3],v[1]/v[3],v[2]/v[3]) for v in verts]
    # print(verts)
    # end experimental part
    if 'h' in flags:
        try:
            ob.hide_set(True)
        except:
            pass
    # Link object to scene and make active
    scn = bpy.context.scene
    scn.collection.objects.link(ob)
    # Create mesh from given verts, faces.
    me.from_pydata(verts, [], faces)
    # Update mesh with new data
    me.update()    
    return ob



def conePrimitive(name,r1,r2,h,wm):
    bpy.ops.mesh.primitive_cone_add(
        radius1 = r1,
        radius2 = r2,
        depth = h,
        location=(0.0, 0.0, 0.0), 
        rotation=(0.0, 0.0, 0.0))
    ob = bpy.context.object
    ob.name = 'turtle_'+name
    ob.matrix_world = wm
    #ob.show_name = True
    me = ob.data
    me.name = name+'Mesh'
    return ob  



def choice(probability):
    if random.random()<=probability:
        return True
    else:
        return False

class Scope():
    
    snaps = {'O':(0,0,0),
        'Z-':(0,0,0.5),
        'Z+':(0,0,-0.5),
        'X-':(0.5,0,0),
        'X+':(-0.5,0,0),
        'Y-':(0,0.5,0),
        'Y+':(0,-0.5,0)}
        
    def __init__(self,w,d,h):
        #self.dims = (w,d,h)
        self.wmatrix = Matrix.Identity(4) @ scale_m((w,d,h))
        
    def scope_dims(self):
        orig_loc, orig_rot, orig_scale = self.wmatrix.decompose()
        return orig_scale
        
    def resize(self,w,d,h,mode='O'):
        (wo,do,ho) = self.scope_dims()
        base = mode.replace('%','')
        if base =='':base = 'O'
        if not('%' in mode):
            # if any of the dims is -1 keep original dim
            (w,d,h) = [dn if dn>=0 else do for dn,do in zip((w,d,h),(wo,do,ho))]
        if base !='O':
            (x,y,z) = (0,0,0)
            #now support multiple plane snaps 
            for i in range(0,len(base),2):
                (xa,ya,za) = Scope.snaps[base[i:i+2]]
                x +=xa
                y +=ya
                z +=za
            if '%' in mode:
                self.translate(x*(w*wo-wo)/wo,y*(d*do-do)/do,z*(h*ho-ho)/ho)
            else:
                self.translate(x*(w-wo)/wo,y*(d-do)/do,z*(h-ho)/ho)
        orig_loc, orig_rot, orig_scale = self.wmatrix.decompose()
        if '%' in mode:
            snew = Vector((w*wo,d*do,h*ho))
        else:
            snew = Vector((w,d,h))
        new_scale = tuple([n/o for o,n in zip(orig_scale,snew)])
        new_scale_mat = Matrix.Scale(new_scale[0],4,(1,0,0)) @ Matrix.Scale(new_scale[1],4,(0,1,0)) @ Matrix.Scale(new_scale[2],4,(0,0,1))
        self.wmatrix = self.wmatrix @ new_scale_mat
    
    def translate(self, x,y,z):
        v3 = (x,y,z)
        self.wmatrix = self.wmatrix @ translate_m(v3)
        
        
    def fit_to(self,wmat):
        self.wmatrix = wmat @ self.wmatrix
  
class Singleton(type):
    _instances = {}
    def __call__(cls,*args,**kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton,cls).__call__(*args, **kwargs)
        return cls.instances[cls]      

class TurtleManager():
    

    
    @classmethod
    def init(cls,seed=0):
        if seed==0:
            random.seed()
        else:
            random.seed(seed)
        cls.caller_stack = []
        cls.geometric_stack = []
        cls.origin = Vector((0.0,0.0,0.0))
        cls.rotation = []
        cls.object_count = 0
        cls.object_limit = 10000
        cls.custom_meshes = {}
        cls.current_scope = Scope(1,1,1)
        cls.subscopes = []
        cls.status = {'deferred':False}
        cls.deferred_instances = []
        cls.recursions ={}
        cls.rnd = random.random()
        cls.RemoveObjects()
        cls.level = 1


    @classmethod    
    def RemoveObjects(cls,mask='turtle'):
        bpy.ops.object.select_all(action='DESELECT')
        for ob in bpy.data.objects:
            if (ob.hide_get() == True) and (mask in ob.name):
                ob.hide_set(False)

        bpy.ops.object.select_pattern(pattern="*"+mask+"*")
        listob = bpy.context.selected_objects
        
        for ob in listob:
            ob.select_set(state=True)
            # mesh = ob.data
            # mesh.user_clear()
            # bpy.data.meshes.remove(mesh)
            # bpy.data.objects.remove(ob,True)
        bpy.ops.object.delete()

        #     mesh = ob.data
        #     mesh.user_clear()
        #     bpy.data.meshes.remove(mesh)

            
        # # deselect all
        # bpy.ops.object.select_all(action='DESELECT')
        # # select existing turtle object
        # bpy.ops.object.select_pattern(pattern="turtle*")
        # # delete them
        # bpy.ops.object.delete()

        for block in bpy.data.meshes:
            if block.users == 0:
                bpy.data.meshes.remove(block)

        for block in bpy.data.materials:
            if block.users == 0:
                bpy.data.materials.remove(block)

        for block in bpy.data.textures:
            if block.users == 0:
                bpy.data.textures.remove(block)

        for block in bpy.data.images:
            if block.users == 0:
                bpy.data.images.remove(block)


    @classmethod
    def t_start(cls):
        cls.geometric_stack.append(copy.deepcopy([cls.current_scope,cls.subscopes,cls.status]))
        cls.subscopes = []
        
    @classmethod
    def t_end(cls):
        geo = copy.deepcopy(cls.geometric_stack.pop())
        cls.current_scope = geo[0]
        cls.subscopes = geo[1]
        cls.status = geo[2]
        
    @classmethod    
    def resize(cls,*args,**kwargs):
        cls.current_scope.resize(*args,**kwargs)
        
    @classmethod
    def dims(cls):
        return cls.current_scope.scope_dims()   
     
    @classmethod
    def printstack(cls):
        print(cls.geometric_stack)
        

    @classmethod    
    def translate(cls, x,y,z):
        #TODO as for rotate, create a version that translate it 'pre-scale'
        v3 = (x,y,z)
        cls.current_scope.wmatrix = cls.current_scope.wmatrix @ translate_m(v3)
        
    @classmethod
    def transform(cls,tmatrix):
        cls.current_scope.wmatrix =  cls.current_scope.wmatrix @ tmatrix
    
    @classmethod
    def set_origin(cls,o3):
        pass
    
    @classmethod
    def deferred(cls,defer=True): #TODO - to be reviewed
        cls.status['deferred'] = defer
    
    @classmethod
    def defer(cls,op):
        cls.deferred_instances.append(['simple_defer',op,cls.current_scope.wmatrix])


    @classmethod
    def if_not_occluded(cls,main_op,occ_tag='',t=[],s=[],alt_op='pass',priority=0):
        if priority == 0:
            priority = cls.level    
        cls.deferred_instances.append([priority,'occlusion_defer',[occ_tag,main_op,t,s,alt_op],cls.current_scope.wmatrix])

    @classmethod
    def run_deferred(cls):
        #if cls.deferred_instances:
            
        while cls.deferred_instances:
            cls.deferred_instances.sort(key=lambda x:x[0])
            [level,type_defer,rule,wm] = cls.deferred_instances.pop(0)
            if type_defer == 'simple_defer':
                cls.current_scope.wmatrix = wm
                cls.apply_op(rule)
            if type_defer == 'occlusion_defer':
                #if it a occlusion defer, then rule is a tuple composed of: 
                #'tag of occluders to test','rule if not occluded','rule if occluded'
                if cls.is_occluded(rule[0],rule[2],rule[3],wm):
                    cls.current_scope.wmatrix = wm
                    cls.apply_op(rule[4])
                else:
                    cls.current_scope.wmatrix = wm
                    cls.apply_op(rule[1])

    @classmethod
    def is_occluded(cls,tag,trans_args,scale_args,wm):
        #create a test cube with matrix wm
        cls.t_start()
        cls.current_scope.wmatrix = wm
        if trans_args:
            cls.translate(*trans_args)
        if scale_args:
            cls.resize(*scale_args)
        cls.cube('test_volume',flags='h')
        tester = bpy.data.objects['turtle_test_volume']
        #make a list of all occluders interested
        list_occluders = [obj for obj in bpy.data.objects if obj.name.startswith('turtle_occluder_'+tag)]
        #test collisions
        collided = False
        while not(collided) and list_occluders:
            tobj = list_occluders.pop(0)
            collided = check_collision(tobj,tester)

        #remove the test cube
        #bpy.data.objects.remove(tester,True)
        if collided:
            tester.name='turtle_collided_tester'
        else:
            tester.name='turtle_used_tester'
        cls.t_end()
        return collided



    @classmethod
    def scale(cls,x,y,z):
        v3 = (x,y,z)
        cls.current_scope.wmatrix = cls.current_scope.wmatrix * scale_m(v3)
        
    @classmethod
    def rotate(cls,rx=0,ry=0,rz=0,inner='O'):
        snaps = {'O':(0,0,0),
            'I':(0,0,0),
            'Z-':(0,0,0.5),
            'Z+':(0,0,-0.5),
            'X-':(0.5,0,0),
            'X+':(-0.5,0,0),
            'Y-':(0,0.5,0),
            'Y+':(0,-0.5,0)}
        if inner == 'I':
            #rotate the instance inside the scope, not rotate the scope
            cls.rotate_inner(rx,ry,rz)
        else:
            #rotate the scope
            (x,y,z) = cls.dims()
            cls.resize(1,1,1)
            cls.rotate_inner(rx,ry,rz)
            cls.resize(x,y,z)
            (dx,dy,dz) = snaps[inner]
            cls.translate(dx,dy,dz)

    @classmethod
    def rotate_inner(cls,rx=0,ry=0,rz=0):
        cls.current_scope.wmatrix = cls.current_scope.wmatrix @  rotate_m(rx,ry,rz)
    
    @classmethod
    def o(cls,olist):
        pass
    
    
    @classmethod
    def occlude(cls,tag='',s=[]):
        cls.t_start()
        if s: cls.resize(*s)
        cls.cube('occluder_'+tag,flags='h')
        cls.t_end()

    @classmethod
    def if_not_occluded_obsolete(main_function,args=[],occ_tag='',alternate='pass'):
        """ a bit more complex implementation
        get the name of the calling function
        register in a list: 
            the occluder tag to check against
            the current scope
            the calling function name
            the alternate function name
        to perform the test later after the main instancing phase, in the deferred phase

        functions are marked as test for occlusion (using this fun call) 
        as a result:
        - the function is not called immediately
        - this method is called instead

        """
        deferred_element = {
                'main_function' : main_function,
                'args' : args,
                'tag' : occ_tag,
                'alternate' : alternate,
                'scope' : copy.deepcopy(cls.current_scope),
                'action' : 'occlusion_test'
        }
        cls.deferred_instances.append(deferred_element)



    @classmethod
    def cube(cls,tag = 'cube',flags=''):
        #cls.called()
        if cls.object_count < cls.object_limit:
            if cls.status['deferred']: #to be removed, as managed separately
                cls.deferred_instances.append(copy.deepcopy(('cubePrimitive',[tag,cls.current_scope.wmatrix])))
            else:
                ob = cubePrimitive(tag,cls.current_scope.wmatrix,flags)
            cls.object_count += 1
        else:
            print('reached limit of: ' + str(cls.object_limit) +' objects')


    @classmethod
    def instanced_primitive(cls,source_object_name,tag = 'cube',flags=''):
        #cls.called()
        if cls.object_count < cls.object_limit:
            if cls.status['deferred']: #to be removed, as managed separately
                cls.deferred_instances.append(copy.deepcopy(('cubePrimitive',[tag,cls.current_scope.wmatrix]))) #TODO align methods
            else:
                ob = instancedPrimitive(tag,source_object_name,cls.current_scope.wmatrix,flags)
            cls.object_count += 1
        else:
            print('reached limit of: ' + str(cls.object_limit) +' objects')
    
    @classmethod        
    def cone(cls,r1 = 1.0, r2=1.0,tag = 'cone'):
        h = 1.0
        #cls.called()
        (x,y,z) = (1,1,1)
        if cls.object_count < cls.object_limit:
            v3 = (x*0.5,y*0.5,z*1.0)
            wm = cls.current_scope.wmatrix @ Matrix.Scale(v3[0],4,(1,0,0)) @ Matrix.Scale(v3[1],4,(0,1,0)) @ Matrix.Scale(v3[2],4,(0,0,1))
            ob = conePrimitive(tag,r1,r2,h,wm)
            
            ob.matrix_world = cls.current_scope.wmatrix @ Matrix.Scale(v3[0],4,(1,0,0)) @ Matrix.Scale(v3[1],4,(0,1,0)) @ Matrix.Scale(v3[2],4,(0,0,1))
            cls.object_count += 1
        else:
            print('reached limit of: ' + cls.object_limit +' objects')   
    
    @classmethod        
    def prism(cls,tx=0.5,ty=0.5,ax=0,ay=0,tag = 'cone'):
        #cls.called()

        (x,y,z) = (1,1,1)
        if cls.object_count < cls.object_limit:
            v3 = (x,y,z)
            wm =  cls.current_scope.wmatrix @ (Matrix.Scale(v3[0],4,(1,0,0)) @ Matrix.Scale(v3[1],4,(0,1,0)) @ Matrix.Scale(v3[2],4,(0,0,1))) 
            ob = prismPrimitive(tag,tx,ty,ax,ay,wm)
            cls.object_count += 1
        else:
            print('reached limit of: ' + cls.object_limit +' objects')   
    
    @classmethod        
    def add_extruded_mesh(cls,name,vlist2,angle=0.0,axis='Z'): #to do: add rotation an centering
        vlist = [(v[0],0,v[1]) for v in vlist2]

        extrusion_vector=(0,1,0)
        #create the mesh
        #scale vertex to be in the 1x1x1 reference cube centered in the origin
        maxd=(-99999,-99999,-99999)
        mind = (99999,99999,99999)
        for v in vlist:
            maxd = (max([v[0],maxd[0]]),max([v[1],maxd[1]]),max([v[2],maxd[2]]))
            mind = (min([v[0],mind[0]]),min([v[1],mind[1]]),min([v[2],mind[2]]))

        trans = tuple([-(j+i)/2 for i,j in zip(mind,maxd)])

        delta = tuple([(j-i) for i,j in zip(mind,maxd)])
        scale = (1/delta[0],1,1/delta[2])
        translation_vector = (trans[0],-0.5,trans[2])
        bm = bmesh.new()
        for v in vlist:
            bm.verts.new(v)
            
        bottom = bm.faces.new(bm.verts)

        top = bmesh.ops.extrude_face_region(bm,geom=[bottom])
        
        bmesh.ops.translate(bm,vec=Vector(extrusion_vector),verts=[v for v in top["geom"] if isinstance(v,bmesh.types.BMVert)])
        #now roto-tranlate the mesh as desired
        bmesh.ops.translate(bm,verts=bm.verts,vec=Vector(translation_vector))
        bmesh.ops.rotate(bm,verts=bm.verts,cent=(0.0,0.0,0.0),matrix=mathutils.Matrix.Rotation(radians(angle), 3, axis))
        bmesh.ops.scale(bm,verts=bm.verts,vec=Vector(scale))
        bm.normal_update()
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me)
        cls.custom_meshes[name] = me.copy()
    
    @classmethod    
    def instance_mesh(cls,name,rx=0,ry=0,rz=0):
        #TODO add deferral logic
        v3 = (1,1,1)
        me = cls.custom_meshes[name].copy()
        ob = bpy.data.objects.new('turtle_'+name,me)
        rot_x = Matrix.Rotation(radians(rx), 4, 'X')
        rot_y = Matrix.Rotation(radians(ry), 4, 'Y')
        rot_z = Matrix.Rotation(radians(rz), 4, 'Z')

        scalem = (Matrix.Scale(v3[0],4,(1,0,0)) @ Matrix.Scale(v3[1],4,(0,1,0)) @ Matrix.Scale(v3[2],4,(0,0,1))  )
        cls.current_scope.wmatrix = cls.current_scope.wmatrix @ scalem
        cls.current_scope.wmatrix = cls.current_scope.wmatrix @ rot_x @ rot_y @ rot_z
        ob.matrix_world =  cls.current_scope.wmatrix 
        cls.object_count += 1
        bpy.context.scene.collection.objects.link(ob)
        bpy.context.view_layer.update()
        
    
    @classmethod    
    def debox(cls,list_trans = ['X+','X-','Y+','Y-']):
        cls.subscopes = []
        subscopes = []
        (x,y,z) = cls.dims()
        #list_trans = ['X+','X-','Y+']
        #list_trans = ['X+','Y+','Z-','Z+']
        trans_dict={
        'X+':[(0.5,0,0),(0.1,y,z),(0,0,0)],
        'X-':[(0.5,0,0),(0.1,y,z),(0,0,180)],
        'Y-':[(0.5,0,0),(0.1,y,z),(0,0,-90)],
        'Y+':[(0.5,0,0),(0.1,y,z),(0,0,90)],
        'Z+':[(0.5,0,0),(0.1,y,z),(0,-90,0)],
        'Z-':[(0.5,0,0),(0.1,y,z),(0,90,0)],
        }
        for i in list_trans:
            with turtle():
                (rx,ry,rz) = trans_dict[i][2]
                cls.rotate(rx,ry,rz,'I')
                #TODO: box depthshould be a param
                cls.resize(0.01,-1,-1,'X+')
                cls.rotate(0,0,90,'I')
                a = copy.deepcopy(cls.current_scope)
                subscopes.append(a)
        cls.subscopes = subscopes
    
    @classmethod      
    def divide(cls,axis='Z',divisions=[1]):
        index_v = { 'X':(1,0,0),
                    'Y':(0,1,0),
                    'Z':(0,0,1)}
        index_d = { 'X':0,
                    'Y':1,
                    'Z':2}
        cls.subscopes = []
        di = divisions
        #preprocess divisions: if any of them is a string prefixed by a, scale them with respect to real dimensions
        fixeds_f = [float(ai[1:]) for ai in di if isinstance(ai,str)]
        variable_r = [ai for ai in di if not(isinstance(ai,str))]
        fixeds_i = [i for ai,i in zip(di,range(0,len(di))) if isinstance(ai,str)]
        if fixeds_f:
            dimref = cls.dims()[index_d[axis]]
            fixdim = sum(fixeds_f)
            reduce_rate = (1-fixdim/dimref)/sum(variable_r)
            fixrates = [fi_f/dimref for fi_f in fixeds_f]
            newdims =[]
            for i in range(0,len(di)):
                if i in fixeds_i:
                    newdims.append(fixrates[fixeds_i.index(i)])
                else:
                    newdims.append(di[i]*reduce_rate)
            di = newdims
                
            
        
        #build working division list
        divs = [0]
        pos = 0
        for el in di:
            pos +=el
            divs.append(pos)

        #split current scope in n subscopes along specified axis

        #compute centers of divisions
        lc = [(j-i)/2+i-0.5 for i,j in zip(divs[:-1],divs[1:])]
        #compute lenght of divisions
        ll = [(j-i) for i,j in zip(divs[:-1],divs[1:])]
        #select scope dimension to split
        #lds = cls.current_scope.dims[index_d[axis]]
        #compute splits and centers in scope dimension
        #lc1 = list(map(lambda x:x*lds-lds/2,lc))
        #ll1 = list(map(lambda x:x*lds,ll))
        #generate subscopes
        subscope_base_dims_v = Vector((1,1,1))
        for scc,scl in zip(lc,ll):
            subscope_base_dims_v[index_d[axis]] = scl
            (w,d,h) = subscope_base_dims_v
            a = Scope(1,1,1)
            trans_v = Vector((0,0,0))
            trans_v[index_d[axis]] = scc
            (x,y,z)=trans_v.to_tuple()
            a.translate(x,y,z)
            a.resize(w,d,h)
            a.fit_to(cls.current_scope.wmatrix)
            #a.ltrans = cls.translate_m(trans_v.to_tuple())
            cls.subscopes.append(a)

    @classmethod
    def deboxed(cls,list_trans = ['X+','X-','Y+','Y-']):
        cls.debox(list_trans)
        for i in range(0,len(cls.subscopes)):
            a = cls.subscopes[i]
            cls.t_start()
            cls.current_scope = a
            yield i
            cls.t_end()

    @classmethod        
    def divided(cls,axis='Z',divisions=[1]):
        cls.divide(axis,divisions)
        for i in range(0,len(cls.subscopes)):
            a = cls.subscopes[i]
            cls.t_start()
            cls.current_scope = a
            yield i
            cls.t_end()

    @classmethod        
    def scope_iter(cls,funct,args):
        getattr(cls,funct)(*args)
        for i in range(0,len(cls.subscopes)):
            a = cls.subscopes[i]
            cls.t_start()
            cls.current_scope = a
            yield i
            cls.t_end()        

    @classmethod  
    def apply_op(cls,ops):
        if not(isinstance(ops,list)):
            ops = [ops]
        for op in ops:
            if isinstance(op,list):
                cls.apply_op(op)
            else:
                ar = ''
                if isinstance(op,tuple):
                    ar = op[1]
                    op = op[0]
                try:
                    getattr(cls,op)(*ar)
                except:
                    try:
                        globals()[op](*ar)
                    except:
                        if op == 'pass':
                            pass
                        else:
                            print('error in calling function: ' + op)
            
    @classmethod
    def div_map(cls,axis='Z',divisions=[1],funlist=[cube]):
        """
        div_map allows to apply multiple operators to a chain of scopes
        now enhanced to support functions with arguments and nested lists of functions
        """
        for el,op in zip(cls.divided(axis,divisions),funlist):
            cls.apply_op(op)
            
#            ar = ''
#            if isinstance(op,tuple):
#                ar = op[1]
#                op = op[0]
#            try:
#                getattr(self,op)(*ar)
#            except:
#                try:
#                    globals()[op](*ar)
#                except:
#                    if op == 'pass':
#                        pass
#                    else:
#                        print('error in calling function: ' + op)



    
def turtled(nr=-1):
    def funturt(func):
        def f_wrapper(*args, **kwargs):
            TurtleManager.t_start()
            TurtleManager.level +=1
            name = func.__name__
            try:
                (rec,maxrec) = t.recursions[name]
            except:
                (rec, maxrec) = (nr,nr)
                TurtleManager.recursions[name] = (rec, maxrec)
            #here add scope moving
            if rec != 0:
                if maxrec >0 : rec = rec -1
                TurtleManager.recursions[name] = (rec, maxrec)
                TurtleManager.rnd = random.random()
                result = func(*args, **kwargs)
            else:
                rec = maxrec
                TurtleManager.recursions[name] = (rec, maxrec)
                result = 0
            TurtleManager.level -=1
            TurtleManager.t_end()
            return result
        return f_wrapper
    return funturt

class turtle():
    def __init__(self,index = -1):
        self.index = index
        pass
    
    def __enter__(self):
        if self.index >= 0:
            #load subscope index
            a = TurtleManager.subscopes[self.index]
            TurtleManager.t_start()
            #a.wmatrix =  a.ltrans
            TurtleManager.current_scope = a
        else:
            TurtleManager.t_start()
    def __exit__(self,type,value,traceback):
        TurtleManager.t_end()
  



@turtled()
def pattern_fill(dir='X',starter='',wstarter='%0.1',list_ele=[],list_w=[],ender='',wender='%0.1',fallback='cube'):
    """
    generic function to fill a scope with a pattern
    starter,ender is the starting and ending element - '' if not to be used
    wstarter,wender is the starting element width - '%dim' if as percentage of total dimension
    list_ele the list of rules of the pattern
    list_w the list of widths of the pattern
    fallback is the fallback rule if the pattern don't fit (for fixed rules)
    """
    tupdims = TurtleManager.dims()
    dirs = {
            'X':0,
            'Y':1,
            'Z':2}
    workdim = tupdims[dirs[dir]]
    list_ele_real = list_ele
    list_w_real = list_w
    #compute starter
    if starter !='':
        if isinstance(wstarter,str):
            wstart_real = workdim*float(wstarter[1:])
        else:
            wstart_real = wstarter
        list_ele_real = [starter]+list_ele_real
        list_w_real = [wstarter] + list_w_real
    #compute ender
    if ender !='':
        if isinstance(wender,str):
            wend_real = workdim*float(wender[1:])
        else:
            wend_real = wender
        list_ele_real = list_ele_real + [ender]
        list_w_real = list_w_real + [wender]
    pattern_w = sum(list_w)
    if (wstart_real + wend_real + pattern_w) <= workdim:
        list_ws = [w/workdim for w in list_w]
        nr_patterns = int((workdim-wstart_real-wend_real)/(pattern_w))
        #rescale starter and ender to fit
        rest_to_fill = (workdim-wstart_real-wend_real-nr_patterns*pattern_w)/workdim
        #print(rest_to_fill)
        srf = rest_to_fill*(wstart_real)/(wstart_real + wend_real)
        erf = rest_to_fill*(wend_real)/(wstart_real + wend_real)
        TurtleManager.div_map(dir,[wstart_real/workdim+srf]+list_ws*nr_patterns+[wend_real/workdim+erf],[starter]+list_ele*nr_patterns+[ender])
        pass
    else:
        if (wstart_real + wend_real) <= workdim:
            wstart1 = wstart_real/(wstart_real + wend_real)
            wend1 = wend_real/(wstart_real + wend_real)
            TurtleManager.div_map(dir,[wstart1,wend1],[starter,ender])
        else:
            TurtleManager.apply_op(fallback)        