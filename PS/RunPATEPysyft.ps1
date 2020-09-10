
$epocss =  @(20,30,40)
$batchsizes =  @( 64,128,256)
$num_teachers = @(20,40, 60, 80)
$learning_rates = @( 0.05, 0.1, 0.2)

$optimizers = @("adam")
$datasets = @("mnist")
$pysyftCmd = "/c C:\Users\vinay.rao\Anaconda3\Scripts\activate.bat C:\Users\vinay.rao\Anaconda3 & conda activate pysyft  & cd /d D:\Vinay\Repos\Differential%20Privacy\py\  & python pysyft_pate.py "


$i=0
function Run($parameters, $config)
{
    foreach($parameter in $parameters)
    {
       
            $file = "D:\Vinay\logs\PATE-pysyft\Log_$($config)-$($parameter)_$($(Get-Date -f 'dd-mm-yyyy-hh-mm-ss')).txt"
            $file = $file -replace '\s',''
        
         
         Write-Host($file)
         $commandDP = "$($pysyftCmd) $($config)=$($parameter)"
         Write-Host($commandDP)
         $procDP = Start-Process cmd  "$($commandDP) >> $($file)" -wait
         write-host("experiment no $($i) :$($config)-$($parameter)")       
         $i++
    }
}


Run $epocss "--epochs" 

Run $batchsizes "--batch_size"

Run $num_teachers  " --num_teachers"

Run $learning_rates  "--learning_rate"

# Run $optimizers  "--optim"

# Run $datasets  "--dataset"
