# automatic-number-plate-recognition
The pipelines detect number plates of all indian standard number plates. It also classifies vehicles into 10 classes as {auto front,auto back, car front, car back, bus back, bus front, truck back, truck front, bike back, bike front}

Four different pipeline have been developed for indian vehicesto to work with different use cases. One works with images of cars, one with pre-recorded video, one works with live cctv footage and one pipeline was developed to benchmark the result on a GPU instance.

## Deployed Link

http://getplates.ml/

## example
### Input 
<img src="https://github.com/akshatkaush/automatic-number-plate-recognition/blob/main/with_photos/test.jpg?raw=true">



### output
<img src="https://github.com/akshatkaush/automatic-number-plate-recognition/blob/main/with_photos/Capture.PNG?raw=true">


There are two weights file, one for lprnet that detcts number plates and one for HRnet segmentation which segments the number plate. They can be availed on request.
