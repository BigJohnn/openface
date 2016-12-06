#!/usr/bin/env th
package.cpath = '/home/hph/.luarocks/lib/lua/5.3/?.so;/usr/local/lib/lua/5.3/?.so;/home/hph/th7/install/lib/?.so;/home/hph/th7/install/lib/lua/5.3/?.so;/home/hph/th7/install/lib/lua/5.3/loadall.so;./?.so'
package.path = '/home/hph/.luarocks/share/lua/5.3/?.lua;/home/hph/.luarocks/share/lua/5.3/?/init.lua;/usr/local/share/lua/5.3/?.lua;/usr/local/share/lua/5.3/?/init.lua;/home/hph/th7/install/share/lua/5.3/?.lua;/home/hph/th7/install/share/lua/5.3/?/init.lua;/home/hph/th7/install/lib/lua/5.3/?.lua;/home/hph/th7/install/lib/lua/5.3/?/init.lua;./?.lua;./?/init.lua'
require 'torch'
require 'optim'

require 'paths'

require 'xlua'
require 'cudnn'

local opts = paths.dofile('opts.lua')

opt = opts.parse(arg)
print(opt)


if opt.cuda then
   require 'cutorch'
   cutorch.setDevice(opt.device)
end

torch.save(paths.concat(opt.save, 'opts.t7'), opt, 'ascii')
print('Saving everything to: ' .. opt.save)

torch.setdefaulttensortype('torch.FloatTensor')

torch.manualSeed(opt.manualSeed)

print('1')
paths.dofile('data.lua')
paths.dofile('util.lua')
model     = nil
criterion = nil
paths.dofile('train.lua')
paths.dofile('test.lua')
print('1')

if opt.peoplePerBatch > nClasses then
  print('\n\nError: opt.peoplePerBatch > number of classes. Please decrease this value.')
  print('  + opt.peoplePerBatch: ', opt.peoplePerBatch)
  print('  + number of classes: ', nClasses)
  os.exit(-1)
end
print('1')

epoch = opt.epochNumber

for _=1,opt.nEpochs do
   train()
   model = saveModel(model)
   if opt.testing then
      test()
   end
   epoch = epoch + 1
end
print('1')
