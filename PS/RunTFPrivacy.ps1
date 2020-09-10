

$batchsizes =  @(  25)
$l2_clips = @(  1.5,2,2.5)
$noises =@(1.3,  1.5,1.7)
$learningRates = @( 0.5)
$epocss =  @(5, 10)

# $epocss =  @(5,10,15,20,30,40,50,60)
# $batchsizes =  @(50, 500,1000)
# $l2_clips = @(  1.5,2,2.5)
# $noises =@(1.3,  1.5,1.7)
# $learningRates = @(0.05, 0.5, 0.35)
$keras = $true

# New-Item $file
if($keras)
{
    $cmd = "/c C:\Users\vinay.rao\Anaconda3\Scripts\activate.bat C:\Users\vinay.rao\Anaconda3 & conda activate primary  & cd /d D:\Vinay\Repos\Differential%20Privacy\Resourses\privacy-masterv2\privacy-master\tutorials  & python .\mnist_dpsgd_tutorial_keras.py  "
}
else {
    $cmd = "/c C:\Users\vinay.rao\Anaconda3\Scripts\activate.bat C:\Users\vinay.rao\Anaconda3 & conda activate primary  & cd /d D:\Vinay\Repos\Differential%20Privacy\Resourses\privacy-master\privacy-master\tutorials  & python .\mnist_dpsgd_tutorial.py  "

    }

          ## --learning_rate=$($defaultLearningRate) --noise_multiplier=$($defaultNoise) --l2_norm_clip=$($defaultL2_clip) --batch_size=$($defaultNBatchsize) 
################################## Epocs##########################
#    foreach($epocs in $epocss)
# {
#     $ proc = Start-Process cmd  "$($cmd) --epochs=$($epocs)" -wait
#     Start-Process cmd  "$($cmd) --epochs=$($epocs) --dpsgd = False " -wait
      
# }
$i=0
function Run($parameters, $config)
{
    foreach($parameter in $parameters)
    {
        if($keras)
        {
            $file = "D:\Vinay\logs\TP\Log_keras_$($config)-$($parameter)_$($(Get-Date -f 'dd-mm-yyyy-hh-mm-ss')).txt"
            $file = $file -replace '\s',''
        }
        else {
            $file = "D:\Vinay\logs\TP\Log_$($config)-$($parameter)_$($(Get-Date -f 'dd-mm-yyyy-hh-mm-ss')).txt"
            $file = $file -replace '\s',''
        }
         
         Write-Host($file)
         $commandDP = "$($cmd) $($config)=$($parameter)"
         Write-Host($commandDP)
         $procDP = Start-Process cmd  "$($commandDP) >> $($file)" -wait
         write-host("experiment no $($i) :$($config)-$($parameter)")       
         $i++
    }
}
# Run $epocss "--nodpsgd --epochs"
# Run $epocss "--epochs" # ran for 5 10 15 20 30


# Run $batchsizes "--batch_size"
# Run $batchsizes "--nodpsgd --batch_size"

# Run $l2_clips  " --l2_norm_clip"
# Run $l2_clips  " --nodpsgd --l2_norm_clip"

# Run $noises  "--noise_multiplier"
# Run $noises  " --nodpsgd --noise_multiplier"

Run $learningRates  "--learning_rate"
Run $learningRates  " --nodpsgd --learning_rate"



################################# batchsizes ##########################
#
#     foreach($batchsize in $batchsizes)
# {
#    Start-Process cmd  "/k C:\Users\vinay.rao\Anaconda3\Scripts\activate.bat C:\Users\vinay.rao\Anaconda3 & activate primary  & cd /d D:\Vinay\Resourses\privacy-master\privacy-master\tutorials &  python .\mnist_dpsgd_tutorial.py --learning_rate=0.15 --noise_multiplier=1.1 --l2_norm_clip=1.0 --batch_size=$batchsize --epochs=60 --data_slice=60000"  
# }




#  python .\mnist_dpsgd_tutorial_keras.py --learning_rate=0.15 --noise_multiplier=1.1
#     --l2_norm_clip =1.0 --batch_size=250 --epochs = $($epocs) --data_slice = 60000 >> D:\Vinay\PS\logs\logs_$(get-date -f "yyyy-MM-dd_hh-mm-ss").txt  "
# & set CUDA_VISIBLE_DEVICES=1