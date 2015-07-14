Chirp.io
========

Chirp is an innovative new platform that allows you to share data using sound. This Python script allows you to convert a url into an audible chirp and vice versa. 

The decoding mechanism scans the recorded audio for any substantial amplitude levels, then a FFT is performed on each segment to find the maximum frequency present. If the frontdoor pair is located then the rest of the chirp is decoded. Once this is complete the url will be opened in your default browser. 

Error correction is currently not implemented. 

Dependencies
------------

- pyAudio
- numpy
- requests

Usage
-----

```
python chirp.py [-h] [-l] [-u <url>] [-c <code>]

optional arguments:
  -h, --help                  show this help message and exit
  -l, --listen	              listen out for chirps
  -u <url>, --url <url>       chirp a url
  -c <code>, --code <code>    chirp a code
```
