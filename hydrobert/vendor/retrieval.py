"""Exact flat squared-L2 adapter, equivalent to FAISS flat search on small banks."""
import types
class ExactL2:
    def __init__(self,*args):self.keys=None
    def reset(self):self.keys=None
    def add(self,keys):self.keys=keys.detach()
    def search(self,queries,k):
        d=(queries.square().sum(-1,keepdim=True)+self.keys.square().sum(-1)[None]-2*queries@self.keys.T).clamp_min(0)
        return d.topk(k,dim=-1,largest=False,sorted=True)
faiss=types.SimpleNamespace(GpuIndexFlatL2=ExactL2,IndexFlatL2=ExactL2,StandardGpuResources=lambda:None)
lib=types.SimpleNamespace(get_d_out=lambda n:1 if n is None else n)
