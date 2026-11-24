"""Matched valve controls, explicit packed storage and a frozen-reader rewrite audit."""
import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from graphnn.model import ModelConfig, JanusModel, _angles
from graphnn.half_memory import HalfEdgeWriter
from graphnn.data import procedural_pairs
from graphnn.rewrite import make_rewrite_batch, fit_rewrite_step, PALETTE

MODES = ("positive", "signed", "binary", "phase", "node", "slots", "binary_slots")
BINARY = {"binary", "binary_slots"}


def pack_bits(x):
    assert x.ndim == 2 and x.shape[1] % 8 == 0
    assert bool(((x == 0) | (x == 1)).all())
    shifts = torch.arange(8, device=x.device, dtype=torch.int64)
    return (x.to(torch.int64).reshape(x.shape[0], -1, 8) << shifts).sum(-1).to(torch.uint8)


def unpack_bits(x):
    shifts = torch.arange(8, device=x.device, dtype=torch.int64)
    return ((x.to(torch.int64)[..., None] >> shifts) & 1).flatten(1).float()


class Memory:
    """Only packed payload persists at evaluation. Float tensor is training-only STE."""
    def __init__(self, values, bits, packed=False):
        self.values, self.bits, self.packed = values, bits, packed

    def flat(self):
        if not self.packed:
            return self.values
        return unpack_bits(self.values) if self.bits == 1 else self.values.float() / 255

    def quantized(self):
        if self.packed:
            return self
        x = self.values.detach().clamp(0, 1)
        p = pack_bits((x >= .5).to(torch.uint8)) if self.bits == 1 else (255*x).round().to(torch.uint8)
        return Memory(p, self.bits, True)


class Valve(nn.Module):
    def __init__(self, mode, cfg=ModelConfig()):
        super().__init__()
        self.mode, self.cfg = mode, cfg
        self.bits = 1 if mode in BINARY else 8
        self.writer = HalfEdgeWriter(cfg)
        self.backbone = JanusModel(cfg)
        self.backbone.writer = nn.Identity()
        self.n = 2*cfg.channels*cfg.grid_size*(cfg.grid_size-1)
        assert self.n % 8 == 0 and cfg.channels % 2 == 0
        self.payload_bytes = self.n*self.bits//8

    def planes(self, x):
        c,g = self.cfg.channels,self.cfg.grid_size
        h,v = x.chunk(2,1)
        return h.reshape(-1,c,g,g-1),v.reshape(-1,c,g-1,g)

    def slot_mask(self, angle, dtype):
        c,g = self.cfg.channels,self.cfg.grid_size
        idx = torch.arange(c,device=angle.device)[None,:]
        mask = (idx >= c//2) == (angle[:,None] > math.pi/4)
        return mask.to(dtype).repeat_interleave(g*(g-1),dim=1).repeat(1,2)

    def write(self, image, context, previous=None):
        angle = _angles(context,len(image),image.device)
        old = image.new_full((len(image),self.n),.5) if previous is None else previous.flat()
        x = self.writer(image,angle,self.planes(old)).flat()
        if self.mode in BINARY:
            hard = (x >= .5).to(x.dtype)
            x = x + (hard-x).detach()
        if self.mode.endswith("slots"):
            # Exact address isolation. Unwritten bits have a fixed initial value.
            if previous is None:
                old = torch.zeros_like(old) if self.bits == 1 else torch.full_like(old,128/255)
            mask = self.slot_mask(angle,x.dtype)
            x = x*mask + old*(1-mask)
        return Memory(x,self.bits)

    def transport(self,r,i,h,v):
        def nbr(a,b,c,d):
            return F.pad(a,(0,1,0,0))+F.pad(b,(1,0,0,0))+F.pad(c,(0,0,0,1))+F.pad(d,(0,0,1,0))
        deg = nbr(torch.ones_like(h),torch.ones_like(h),torch.ones_like(v),torch.ones_like(v))
        def scalar(z,wh,wv):
            return nbr(wh*z[...,1:],wh*z[...,:-1],wv*z[...,1:,:],wv*z[...,:-1,:])
        if self.mode == "phase":
            hc,vc = torch.polar(torch.ones_like(h),(2*h-1)*math.pi),torch.polar(torch.ones_like(v),(2*v-1)*math.pi)
            z = torch.complex(r,i)
            y = .5*z+.5*nbr(hc*z[...,1:],hc.conj()*z[...,:-1],vc*z[...,1:,:],vc.conj()*z[...,:-1,:])/deg
            return y.real,y.imag
        wh,wv = (h,v) if self.mode=="positive" else (2*h-1,2*v-1)
        if self.mode == "node":
            # Node-source gates, one for horizontal and one for vertical outgoing messages.
            # Same gate acts on BOTH components, unlike Neon's earlier node control.
            nh,nv = F.pad(wh,(0,1,0,0)),F.pad(wv,(0,0,0,1))
            def node(z):
                a,b = z*nh,z*nv
                return nbr(a[...,1:],a[...,:-1],b[...,1:,:],b[...,:-1,:])
            return .5*r+.5*node(r)/deg,.5*i+.5*node(i)/deg
        return .5*r+.5*scalar(r,wh,wv)/deg,.5*i+.5*scalar(i,wh,wv)/deg

    def read_flat(self,x,angle):
        theta = _angles(angle,len(x),x.device)
        z = self.backbone._fresh_ping(theta)
        r,i = z.real,z.imag
        if self.mode.endswith("slots"):
            mask = self.slot_mask(theta,x.dtype)
            h,v = self.planes((2*x-1)*mask)
            h,v = F.pad(h,(0,1,0,0)),F.pad(v,(0,0,0,1))
            features = [r,i,h,v,r*h,i*v]
        else:
            h,v = self.planes(x)
            features = [r,i]
            for j in range(1,self.cfg.steps+1):
                r,i = self.transport(r,i,h,v)
                if j in (self.cfg.steps//2,self.cfg.steps):
                    features.extend((r,i))
        return self.backbone.decoder(F.interpolate(torch.cat(features,1),size=(self.cfg.image_size,)*2,mode="bilinear",align_corners=False))

    def read(self,m,angle):
        return self.read_flat(m.flat(),angle)


def mse(a,b):
    return float((a-b).square().mean())


def hit(pred,mask,label):
    color = (pred*mask).sum((-2,-1))/mask.sum((-2,-1)).clamp_min(1)
    pal = pred.new_tensor(PALETTE)
    return ((color[:,None,:]-pal[None,:,:]).square().sum(-1).argmin(-1)==label)


@torch.no_grad()
def evaluate(model,batch,fresh):
    model.eval()
    a,b = batch.original.unbind(1)
    ap = batch.updated[:,0]
    ma,mb = batch.masks.unbind(1)
    la,lb = batch.labels.unbind(1)
    bp = b*(1-mb)+mb*b.new_tensor(PALETTE)[1-lb,:,None,None]
    a_new = fresh.original[:,0]
    s = model.write(b,90.,model.write(a,0.).quantized()).quantized()
    oldb = model.read(s,90.)
    olda = model.read(s,0.)
    def write(x,angle,m):
        p = model.write(x,angle,m).quantized()
        assert p.values.dtype==torch.uint8 and p.values.shape[1]==model.payload_bytes
        return p
    def scores(m,ta,tb,maska,maskb,la,lb):
        pa,pb = model.read(m,0.),model.read(m,90.)
        return {"joint_mark":float((hit(pa,maska,la)&hit(pb,maskb,lb)).float().mean()),
                "a_mse":mse(pa,ta),"b_mse":mse(pb,tb)}
    rev = write(ap,0.,s)
    sham = write(a,0.,s)
    swap = write(bp,90.,s)
    replacement = write(a_new,0.,s)
    chain = write(a,0.,write(bp,90.,rev))
    out = {"id":scores(rev,ap,b,ma,mb,1-la,lb),
           "swap":scores(swap,a,bp,ma,mb,la,1-lb),
           "replace":scores(replacement,a_new,b,fresh.masks[:,0],mb,fresh.labels[:,0],lb),
           "chain":scores(chain,a,bp,ma,mb,la,1-lb),
           "b_drift":mse(model.read(rev,90.),oldb),
           "swap_a_drift":mse(model.read(swap,0.),olda),
           "sham_b_drift":mse(model.read(sham,90.),oldb),
           "payload_bytes":model.payload_bytes,
           "shared_params":sum(p.numel() for p in model.parameters()),
           "shared_weight_bytes":sum(p.numel()*p.element_size() for p in model.parameters())}
    for name,m in (("rewrite",rev),("sham",sham)):
        d=m.flat()-s.flat()
        out[name+"_changed_value_fraction"]=float((d!=0).float().mean())
        out[name+"_changed_packed_byte_fraction"]=float((m.values!=s.values).float().mean())
        out[name+"_mean_abs_value_change"]=float(d.abs().mean())
        out[name+"_rms_value_change"]=float(d.square().mean().sqrt())
    return out,(s,rev,sham)


def geometry(model,states,seed,count=8,random_trials=4):
    """Float-reader directional derivatives, plus exact discrete matched null.
    JVP is a smooth extension for binary memory, NOT a physical infinitesimal bit flip.
    Random local null uses per-sample signed permutation of the same update magnitudes.
    Discrete null permutes actual new values among valid changed locations of the same
    original value (binary: flips equal counts of zeros and ones at random locations).
    For 8-bit memory the finite null uses clipped perturbations and reports actual norm.
    """
    if count==0:
        return {}
    model.eval()
    x,real,sham = [m.flat()[:count].detach() for m in states]
    rng=torch.Generator(device=x.device).manual_seed(seed+90001)
    def summarize(d):
        with torch.enable_grad():
            a=torch.autograd.functional.jvp(lambda t:model.read_flat(t,0.),x,d,create_graph=False)[1]
            b=torch.autograd.functional.jvp(lambda t:model.read_flat(t,90.),x,d,create_graph=False)[1]
        ar=a.flatten(1).square().mean(1)
        br=b.flatten(1).square().mean(1)
        nr=d.square().mean(1)
        return {"a_jvp_mse":float(ar.mean()),"b_jvp_mse":float(br.mean()),
                "delta_rms":float(nr.mean().sqrt()),
                "b_jvp_per_delta_energy":float(br.sum()/nr.sum().clamp_min(1e-12))}
    @torch.no_grad()
    def finite(y):
        return {"a_drift":mse(model.read_flat(y,0.),model.read_flat(x,0.)),
                "b_drift":mse(model.read_flat(y,90.),model.read_flat(x,90.)),
                "delta_rms":float((y-x).square().mean().sqrt())}
    result={}
    for name,y in (("rewrite",real),("sham",sham)):
        d=y-x
        trial=[]
        for _ in range(random_trials):
            order=torch.rand(x.shape,device=x.device,generator=rng).argsort(1)
            signs=torch.where(torch.rand(x.shape,device=x.device,generator=rng)<.5,-1.,1.)
            rand=d.gather(1,order)*signs
            if model.bits==1:
                null=x.clone()
                for row in range(len(x)):
                    for val in (0.,1.):
                        k=int(((x[row]==val)&(d[row]!=0)).sum())
                        ix=torch.where(x[row]==val)[0]
                        ix=ix[torch.randperm(len(ix),device=x.device,generator=rng)[:k]]
                        null[row,ix]=1-val
            else:
                null=((x+rand).clamp(0,1)*255).round()/255
            trial.append({"local":summarize(rand),"finite":finite(null)})
        result[name]={"local":summarize(d),"finite":finite(y),"random":trial}
    return result


def self_test():
    torch.set_num_threads(1)
    x=torch.randint(0,2,(3,896)).float()
    assert pack_bits(x).shape==(3,112) and torch.equal(x,unpack_bits(pack_bits(x)))
    batch=make_rewrite_batch(procedural_pairs(2))
    models={m:Valve(m) for m in MODES}
    assert len({sum(p.numel() for p in m.parameters()) for m in models.values()})==1
    opts={m:torch.optim.Adam(v.parameters(),lr=.003) for m,v in models.items()}
    fit_rewrite_step(models,opts,batch)
    for name,m in models.items():
        out,states=evaluate(m,batch,batch)
        assert out["payload_bytes"]==(112 if name in BINARY else 896)
        assert math.isfinite(out["id"]["a_mse"])
        if name.endswith("slots"):
            assert out["b_drift"]==0 and out["swap_a_drift"]==0
        if name in ("signed","binary"):
            audit=geometry(m,states,41,count=1,random_trials=1)
            assert math.isfinite(audit["rewrite"]["local"]["b_jvp_mse"])
    print("PASS: packing, training, evaluation, slots, Jacobian probe (all seven modes)")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--steps",type=int,default=500)
    p.add_argument("--seeds",type=int,nargs="+",default=[41,97,127])
    p.add_argument("--modes",nargs="+",choices=MODES,default=list(MODES))
    p.add_argument("--train-count",type=int,default=128)
    p.add_argument("--test-count",type=int,default=128)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--geometry-count",type=int,default=8)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output",type=Path,default=Path("runs/gate1"))
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:
        self_test()
        return
    if min(args.steps,args.batch_size)<1 or min(args.train_count,args.test_count)<2 or args.geometry_count<0:
        p.error("invalid experiment size")
    if args.device=="cpu":
        torch.set_num_threads(min(4,torch.get_num_threads()))
    args.output.mkdir(parents=True,exist_ok=True)
    all_results={}
    start=time.perf_counter()
    def save():
        report={"config":{**vars(args),"output":str(args.output)},"elapsed_seconds":time.perf_counter()-start,"per_seed":all_results}
        keys=["b_drift","swap_a_drift","sham_b_drift","rewrite_rms_value_change","sham_rms_value_change"]
        report["means"]={}
        for mode in args.modes:
            rows=[v[mode] for v in all_results.values() if mode in v]
            if not rows: continue
            report["means"][mode]={"seeds_completed":len(rows),"payload_bytes":rows[0]["payload_bytes"],
                **{k:sum(v[k] for v in rows)/len(rows) for k in keys},
                **{s:{k:sum(v[s][k] for v in rows)/len(rows) for k in rows[0][s]} for s in ("id","swap","replace","chain")}}
        (args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
        return report
    for seed in args.seeds:
        train=make_rewrite_batch(procedural_pairs(args.train_count,seed=seed),seed+1000)
        test=make_rewrite_batch(procedural_pairs(args.test_count,seed=seed+10000),seed+20000).to(args.device)
        fresh=make_rewrite_batch(procedural_pairs(args.test_count,seed=seed+30000),seed+40000).to(args.device)
        # Fixed, shared minibatch sequence across arms, no test data in training.
        rng=torch.Generator().manual_seed(seed)
        batches=[torch.randint(args.train_count,(args.batch_size,),generator=rng) for _ in range(args.steps)]
        all_results[str(seed)]={}
        for mode in args.modes:
            torch.manual_seed(seed)
            model=Valve(mode).to(args.device)
            opt=torch.optim.Adam(model.parameters(),lr=.003)
            for step,ids in enumerate(batches,1):
                loss=fit_rewrite_step({mode:model},{mode:opt},train.select(ids).to(args.device))[mode]
                if step==1 or step%100==0 or step==args.steps:
                    print(f"{seed} {mode} {step}/{args.steps} loss={loss:.5f}",flush=True)
            out,states=evaluate(model,test,fresh)
            out["geometry"]=geometry(model,states,seed,args.geometry_count)
            all_results[str(seed)][mode]=out
            torch.save({"mode":mode,"state_dict":model.state_dict()},args.output/f"{seed}_{mode}.pt")
            save()
            print(f"{seed} {mode}: {out['payload_bytes']} bytes, chain {out['chain']}",flush=True)
    print(json.dumps(save()["means"],indent=2))


if __name__=="__main__":
    main()
