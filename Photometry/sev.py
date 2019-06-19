import os
import re
import struct

import numpy as np

# import SEV file into numpy array
def read_sev(sev_file):

    if os.path.isfile(sev_file):
        fileName, fileExtension = os.path.splitext(sev_file)
        if fileExtension.lower() != '.sev':
            print('not a valid SEV file')
            return
        print('importing data from single SEV file:', sev_file)
    else:
        print('not a valid file')
        return None

    ALLOWED_FORMATS = [np.float32, np.int32, np.int16, np.int8, np.float64, np.int64]        

    f = open(sev_file, 'rb')
    head = f.read(40)
        
    fileSizeBytes, fileType, fileVersion, eventName, channelNum, totalNumChannels, sampleWidthBytes, reserved1, dForm, decimate, rate, reserved2 = struct.unpack('Q3sB4sHHHHBBHQ',head)

    if fileType.lower() != 'sev':
        print('bad file type', fileType)
        return None
        
    if fileVersion > 2:
        print('unknown version:', fileVersion)
        return None
        
    if fileVersion < 2:
        eventName = eventName[::-1]

    if fileVersion > 0:
        # determine data sampling rate
        fs = 2.**(float(rate))*25000000./2.**12./float(decimate)
    else:
        dForm = 0
        fs = 0
        parts = sev_file.split('_')
        eventName = parts[-2]
        channelNum = int(re.search(r'\d+', parts[-1]).group())
        print('Warning - empty header; assuming {0} store, Ch {1} format {2} and fs = {3}\nupgrade to OpenEx v2.18 or above for proper header information\n'.format(eventName,  channelNum, dForm, 24414.0625))
        
        fs = 24414.0625

    fmt = ALLOWED_FORMATS[dForm & 7]
    
    d = dict()
    d['fileSizeBytes']    = fileSizeBytes
    d['fileType']         = fileType
    d['fileVersion']      = fileVersion
    d['eventName']        = eventName
    d['channelNum']       = channelNum
    d['totalNumChannels'] = totalNumChannels
    d['sampleWidthBytes'] = sampleWidthBytes
    d['dForm']            = dForm
    d['decimate']         = decimate
    d['rate']             = rate
    d['fs']               = fs
    d['fmt']              = fmt
    d['data']             = np.fromfile(f, dtype=fmt)
    
    if fileVersion > 0:
        # verify streamHeader is 40 bytes
        streamHeaderSizeBytes = d['fileSizeBytes'] - len(d['data']) * d['sampleWidthBytes'];
        if streamHeaderSizeBytes != 40:
            print('Warning - Header Size Mismatch -- {0} bytes vs 40 bytes'.format(streamHeaderSizeBytes))
            
    return d








if __name__ == '__main__':
    sev_file = '/path/to/sev'
    d = read_sev(sev_file)
    print(d)


