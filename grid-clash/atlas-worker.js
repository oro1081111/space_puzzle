// Fixed-version runtime; all inference stays on the player's device.
importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.wasm.min.js');
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
const sessions=new Map();
self.onmessage = async ({data}) => {
  try {
    const {board, positions, blocks, turn} = data.state;
    const n=board.length,total=n*n;
    if(![15,30].includes(n)||positions.length!==2)throw new Error('ATLAS-R supports 15/30 two-player games only');
    if(!sessions.has(n))sessions.set(n,await ort.InferenceSession.create(n===15?'atlas-r-v1.onnx':'atlas-r-v1-30.onnx', {executionProviders:['wasm']}));
    const session=sessions.get(n);
    const input = new Float32Array(2*7*total);
    for(let owner=0;owner<2;owner++){
      const base=owner*7*total;
      for(let y=0;y<n;y++)for(let x=0;x<n;x++){
        const z=y*n+x, v=board[y][x];
        input[base+(v===owner?0:v===-1?2:1)*total+z]=1;
        input[base+6*total+z]=Math.max(0,1-turn/(total*8+1));
      }
      for(let c=0;c<2;c++){
        const [x,y]=positions[c===0?owner:1-owner];input[base+(3+c)*total+y*n+x]=1;
      }
      for(const [x,y] of blocks)input[base+5*total+y*n+x]=1;
    }
    const result=await session.run({board:new ort.Tensor('float32',input,[2,7,n,n])});
    self.postMessage({id:data.id,logits:Array.from(result.logits.data)});
  }catch(error){self.postMessage({id:data.id,error:String(error)});}
};
