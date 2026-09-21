// Fixed-version runtime; all inference stays on the player's device.
importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.wasm.min.js');
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
const sessions=new Map();
self.onmessage = async ({data}) => {
  try {
    const {board, positions, blocks, turn} = data.state;
    const n=board.length,total=n*n;
    if(![15,30].includes(n)||positions.length<2||positions.length>8)throw new Error('ATLAS-R supports 15/30 boards and 2..8 players');
    const model=n===15&&positions.length===3?'atlas-r-3p-v1.onnx':n===15?'atlas-r-v1.onnx':'atlas-r-v1-30.onnx';
    if(!sessions.has(model))sessions.set(model,await ort.InferenceSession.create(model, {executionProviders:['wasm']}));
    const session=sessions.get(model);
    const logits=[];
    // Keep the frozen batch-two model: evaluate all players in pairs.
    for(let pair=0;pair<positions.length;pair+=2){
     const input = new Float32Array(2*7*total);
     for(let slot=0;slot<2;slot++){
      const owner=Math.min(pair+slot,positions.length-1),base=slot*7*total;
      for(let y=0;y<n;y++)for(let x=0;x<n;x++){
        const z=y*n+x, v=board[y][x];
        input[base+(v===owner?0:v===-1?2:1)*total+z]=1;
        input[base+6*total+z]=Math.max(0,1-turn/(total*8+1));
      }
      for(let other=0;other<positions.length;other++){
        const [x,y]=positions[other];input[base+(other===owner?3:4)*total+y*n+x]=1;
      }
      for(const [x,y] of blocks)input[base+5*total+y*n+x]=1;
    }
    const result=await session.run({board:new ort.Tensor('float32',input,[2,7,n,n])});
    logits.push(...Array.from(result.logits.data).slice(0,Math.min(2,positions.length-pair)*4));
    }
    self.postMessage({id:data.id,logits});
  }catch(error){self.postMessage({id:data.id,error:String(error)});}
};
