local M = { }

-- http://stackoverflow.com/questions/6380820/get-:containing-path-of-lua-file
function script_path()
   local str = debug.getinfo(2, "S").source:sub(2)
   return str:match("(.*/)")
end

function M.parse(arg)

   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('OpenFace')
   cmd:text()
   cmd:text('Options:')

   ------------ General options --------------------
   cmd:option('-cache',
              paths.concat(script_path(), 'work'),
              'Directory to cache experiments and data.')
   cmd:option('-save', paths.concat(script_path(), 'experiment'), 'Directory to save experiment.')
   cmd:option('-data',
              paths.concat(os.getenv('HOME'), 'openface', 'data', 'vgg', 'aligned224'),
                           -- 'dlib-affine-224-split'),
              'Home of dataset. Images separated by identity.')
   cmd:option('-manualSeed', 5, 'Manually set RNG seed')
   cmd:option('-cuda', true, 'Use cuda.')
   cmd:option('-device', 2, 'Cuda device to use.')
  -- cmd:option('-device', 2, 'Cuda device to use.')
   cmd:option('-nGPU',   1,  'Number of GPUs to use by default')
   cmd:option('-cudnn', true, 'Convert the model to cudnn.')
   cmd:option('-cudnn_bench', true, 'Run cudnn to choose fastest option. Increase memory usage')

   ------------- Data options ------------------------
   cmd:option('-nDonkeys', 1, 'number of donkeys to initialize (data loading threads)')

   ------------- Training options --------------------
   cmd:option('-nEpochs', 1000, 'Number of total epochs to run')
   cmd:option('-epochSize', 50, 'Number of batches per epoch')
   cmd:option('-epochNumber', 1, 'Manual epoch number (useful on restarts)')
   -- GPU memory usage depends on peoplePerBatch and imagesPerPerson.
   cmd:option('-peoplePerBatch', 5, 'Number of people to sample in each mini-batch.') --default is 15
   cmd:option('-imagesPerPerson',15, 'Number of images to sample per person in each mini-batch.')
   cmd:option('-testing', false, 'Test with the LFW.')
   cmd:option('-testBatchSize', 800, 'Batch size for testing.')
   cmd:option('-lfwDir', '../data/lfw/align', 'LFW aligned image directory for testing.')

   ---------- Model options ----------------------------------
   cmd:option('-retrain', 'none', 'provide path to model to retrain with')
   cmd:option('-modelDef', '../models/openface/nn2.def.lua', 'path to model definiton')
   cmd:option('-imgDim', 224, 'Image dimension. nn2=224, nn4=96')
   cmd:option('-embSize', 128, 'size of embedding from model')
   cmd:option('-alpha', 0.2, 'margin in TripletLoss')
   cmd:text()

   local opt = cmd:parse(arg or {})
   os.execute('mkdir -p ' .. opt.cache)

   if opt.save == '' then
      opt.save = paths.concat(opt.cache, os.date("%Y-%m-%d_%H-%M-%S"))
   end
   os.execute('mkdir -p ' .. opt.save)

   return opt
end

return M
