import mlx.core as mx
mx.random.seed(1)
def bits(a): return a.view(mx.uint32) if a.dtype==mx.float32 else a.view(mx.uint16)
def same(a,b): return a.shape==b.shape and bool(mx.all(bits(a)==bits(b)).item())
for (N,K,dt) in [(128,2304,mx.float32),(2304,64,mx.float32),(64,4608,mx.bfloat16),(201024,64,mx.float32)]:
    W=(mx.random.normal((N,K))*0.05).astype(dt)
    for C in [2,3,4,8]:
        x=mx.random.normal((C,1,K)).astype(dt)
        one=mx.concatenate([x[i:i+1]@W.T for i in range(C)])
        r={}
        r['bcast0']=x@mx.broadcast_to(W.T[None],(C,K,N))
        r['gather_mm']=mx.gather_mm(x,W.T[None],rhs_indices=mx.zeros((C,),mx.uint32))
        r['gather_mm_l']=mx.gather_mm(x,W.T[None],lhs_indices=mx.arange(C).astype(mx.uint32),rhs_indices=mx.zeros((C,),mx.uint32))
        r['matvec_T']=(W@x.reshape(C,K).T).T.reshape(C,1,N)
        r['einsum']=mx.einsum('cik,nk->cin',x,W)
        r['addmm']=mx.addmm(mx.zeros((C,1,N),dt),x,W.T)
        mx.eval(one,*r.values())
        print(N,K,dt,C," ".join("%s:%s"%(k,'ok' if same(one,v) else ('DIFF' if one.shape==v.shape else 'SHAPE%s'%(v.shape,))) for k,v in r.items()),flush=True)
