Chirp.io
========

Chirp is an innovative new platform that allows you to share data using sound. This Python script allows you to convert a url into an audible chirp and vice versa. As the API is not officially released yet, the decoding function only returns the chirp code. Error correction is currently not implemented. 

Dependencies
------------

- pyAudio
- numpy
- requests

Usage
-----

```
python chirp.py [-h] [-l <n>] [-u <url>] [-c <code>]

optional arguments:
  -h, --help                  show this help message and exit
  -l <n>, --listen <n>        listen for a chirp for n seconds
  -u <url>, --url <url>       chirp a url
  -c <code>, --code <code>    chirp a code
```
