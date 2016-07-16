Chirp.io
========

Chirp is an interesting new platform that allows you to share data using sound. This Python script allows you to convert a link, image or pdf into an audible chirp and vice versa.

The decoding mechanism scans the recorded audio for any substantial amplitude levels, then a FFT is performed on each segment to find the maximum frequency present. If the frontdoor pair is located then the rest of the chirp is decoded. Once this is complete the url will be opened in your default browser.

For encoding, a sine wave is generated for each character in the chirp message.

For error correction I am using [Tomer Filiba's](https://github.com/tomerfiliba/reedsolomon) Reed Solomon implementation. This can correct up to four errors, however this only works if using this script to both send and receive the chirps. If anyone knows how to update this to produce the same error characters as the Chirp API, I would be very interested to find out.

Dependencies
------------

- pyAudio
- numpy
- requests
- python-magic (libmagic)

Usage
-----

```
python chirp.py [-h] [-l] [-i] [-u <url>] [-c <code>] [-f <file>]

optional arguments:
  -h, --help                  show this help message and exit
  -l, --listen	              listen out for chirps
  -i, --internal              uses internal error correction
  -u <url>, --url <url>       chirp a url
  -c <code>, --code <code>    chirp a code
  -f <file>, --file <file>    chirp a file, path to either a jpg, png or pdf
```
