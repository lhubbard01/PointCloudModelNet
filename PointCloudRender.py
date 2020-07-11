class CloudRender:
  def __init__(self, path):
    self.path = path
    self.tris = None
    points, indices, vertices = self.load_cloud()
    self.points = points
    self.indices = indices
    self.vertices = vertices


  def load_cloud(self):
    with open(self.path, "r") as f:
      x=f.readlines()
    counts = list(map(int, x[1].strip().split(" ")))
    points = np.array([list(map(float, x[i].strip().split(" "))) for i in range(2,counts[0]+2)])
    indices = [list(map(int, x[i].strip().split(" ")[1:])) for i in range(2+counts[0], counts[1]+2+counts[0])]
    return points, indices, map_vertices(points, indices)


  
  def map_vertices(self, points, connections):
    verts = []
    for i in range(len(connections)):
      triangle = connections[i]
      verts.append([points[triangle[0]], points[triangle[1]], points[triangle[2]]])
    return np.array(verts)



  def show_cloud(self):
    fig, ax = plt.subplots(figsize=((15,9)),
                          subplot_kw={"projection":"3d", "elev":0})
    ax.scatter(self.points[:,0], self.points[:,1], self.points[:,2])
    return fig

  def show_all(self,elev=0.0,azimuth=0.0):
    if self.tris is None:
      self.tris = mtri.Triangulation(x=self.points[:,0],y=self.points[:,1],triangles=self.indices)
    fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=((15,15)), subplot_kw={"projection":"3d", "elev":elev,"azim":azimuth})
    ax1.plot_trisurf(self.tris,self.points[:,2])
    ax2.scatter(self.points[:,0], self.points[:,1], self.points[:,2])
    return fig

  def show_verts(self,):
    fig, ax = plt.subplots(figsize=((15,9)),subplot_kw={"projection":"3d", "elev":0})
    if self.tris is None:
      self.tris = mtri.Triangulation(x=self.points[:,0],y=self.points[:,1],triangles=self.indices)
    ax.plot_trisurf(self.tris,self.points[:,2])
    return fig