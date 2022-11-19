from DetectorConfig     import DetectorConfig
from AsyncBuffer        import AsyncBuffer
from PulseStats         import PulseStats
from Waveform           import Waveform
from Threshold          import Threshold
from CSingleUDPReceiver import CSingleUDPReceiver

import socket
import math
import numpy as np
from pytictoc import TicToc

# function [] = updatebufferreadvariables(ps_input)
def updateBufferReadVariables(Config: DetectorConfig, ps_input: PulseStats, stftOverlapFraction: float):
    # This function updates the buffer reading variables as needed by
    # the priori information about the pulse statistics. The segment
    # lengths depend on the interpulse duration and uncertainty
    # parameters.

    # Build an empty waveform just so that we can calculate number
    # of overlapSamples. This is needed for buffer operations

    # X0 = waveform(single(complex([])), Config.Fs, 0, ps_input, stftOverlapFraction, threshold(0.01));
    X0 = Waveform(None, Config.Fs, 0, ps_input, stftOverlapFraction, Threshold(0.01));

    X0.setPrioriDependentProps(ps_input)

    # [~,~,n_ol,n_ws,~,~,N,M,J] = X0.getprioridependentprops(X0.ps_pre);
    [_, _, n_ol, n_ws, _, _, N, M, J] = X0.getPrioriDependentProps(X0.ps_pre);
    
    #          overlapWindows  = 2*Config.K*M+1;
    #          overlapSamples	= n_ws*overlapWindows;
    #          # sampsForKPulses = Config.K*n_ws*(N+M+1+1);
    #          sampsForKPulses = n_ws*(Config.K*(N+M)+1+1);

    overlapWindows  = 2 * (Config.K * M + J);
    overlapSamples	= n_ws * overlapWindows;
    #          if Config.K~=1
    #              sampsForKPulses = n_ws*(Config.K*(N+M)-2*M)+n_ol;
    #          else
    #              sampsForKPulses = n_ws*(N+M+J)+n_ol;
    #          end
    # See 2022-07-11 for updates to samples def
    sampsForKPulses = n_ws * (Config.K * (N + M) + J + 1) + n_ol;

    print('Updating buffer read vars|| N: %d u, M: %d u, J: %d u' % (int(N), int(M), int(J)))
    print('Updating buffer read vars|| sampForKPulses: %d u,  overlapSamples: %d u,' % (int(sampsForKPulses), int(overlapSamples)))

    return sampsForKPulses, overlapSamples

# function [] = uavrt_detection()
def uavrt_detection():
    Config =  DetectorConfig();
    print(Config)

    # Initialize states for operational modes
    if Config.opMode == 'freqSearchHardLock':
        fLock = False;
    elif Config.opMode == 'freqKnownHardLock':
        fLock = True;
    elif Config.opMode == 'freqSearchSoftLock':
        fLock = False;
    elif Config.opMode == 'freqAllNoLock':
        fLock = False;
    else:
        fLock = False;

    prioriRelativeFreqHz = 10e-6 * abs(Config.tagFreqMHz - Config.channelCenterFreqMHz);
    ps_pre = PulseStats(t_p = Config.tp, t_ip = Config.tip, t_ipu = Config.tipu, t_ipj = Config.tipj, fp = prioriRelativeFreqHz)

    stftOverlapFraction = 0.5;
    zetas               = [ 0, 0.5 ];
    pauseWhenIdleTime   = 0.25;

    # Initialize and then set these variable needed for buffer reads
    sampsForKPulses, overlapSamples = updateBufferReadVariables(Config, ps_pre, stftOverlapFraction);

    packetLength = 1025; # 1024 plus a time stamp.
    print('Startup set 1 complete.')

    # #  Prepare data writing buffer
    #  Calculating the max size that would ever be needed for the buffer
    #  maxK    = 6;
    #  maxFs   = 912000/2;
    #  maxtp   = 0.04;
    #  maxtip  = 2;
    #  maxtipu = 0.1;
    #  maxpulseStatsPriori = pulsestats(maxtp,maxtip,maxtipu,[],[],[],[1 1],pulse);
    #  Xmax = waveform([], maxFs, 0, maxpulseStatsPriori, stftOverlapFraction);
    #  [~,~,~,maxn_ws,~,~,maxN,maxM] = Xmax.getprioridependentprops(Xmax.ps_pre);
    #  sampsForMaxPulses = maxK*maxn_ws*(maxN+maxM+1+1);
    sampsForMaxPulses = 24810 * 2;
    asyncDataBuff = AsyncBuffer(sampsForMaxPulses, np.csingle);
    asyncTimeBuff = AsyncBuffer(sampsForMaxPulses, float);
    print('Startup set 2 complete')
    dataWriterTimeIntervalNominal   = 10; # Write interval in seconds. 2.5*60*4000*32/8 should work out the 2.4Mb of data at 4ksps.
    dataWriterPacketsPerInterval    = math.ceil(dataWriterTimeIntervalNominal / ((packetLength - 1) / Config.Fs));
    dataWriterTimeIntervalActual    = dataWriterPacketsPerInterval * packetLength / Config.Fs;
    dataWriterSamples               = dataWriterPacketsPerInterval * packetLength;

    try:
        dataWriterFileID = open(Config.dataRecordPath, 'w');
    except:
        print("UAV-RT: Error opening/creating data record file with error:")
        dataWriterFileID = None

    print('Startup set 3 complete.')

    print('Startup set 5 complete.')

    print('Startup set 6 complete.')

    samplesPerFrame = 2048;
    udpReceiver = CSingleUDPReceiver(Config.ipData, Config.portData);

    # Initialize loop variables
    resetBuffersFlag  = True;
    framesReceived    = 0;
    segmentsProcessed = 0;
    suggestedMode     = 'S';
    fLock             = False;
    staleDataFlag     = True;   # Force buffer  flush on start
    idleTic           = 1;
    i                 = 1;
    timeStamp         = 0;
    lastTimeStamp     = 0;
    cleanBuffer       = True;
    trackedCount      = 0;

    tictoc = TicToc();

    print('Startup set 7 complete. Starting processing...')

    while True:
        if resetBuffersFlag:
            asyncDataBuff.reset();
            asyncTimeBuff.reset();
            resetBuffersFlag    = False;
            cleanBuffer         = True;

        # #  Flush UDP buffer if data in the buffer is stale.
        if staleDataFlag:
            print('********STALE DATA FLAG: %s *********' %  staleDataFlag);
            udpReceiver.clear()
            staleDataFlag = False;

        iqData  = udpReceiver.read(samplesPerFrame * 2);

        framesReceived = framesReceived + 1;
        # timeVector     = timeStamp+1/Config.Fs*(0:(numel(iqData)-1)).';
        # timeStamp      = timeStamp + (numel(iqData) / Config.Fs);
        timeVector     = timeStamp + 1 / Config.Fs * np.arange(len(iqData));
        timeStamp      = timeStamp + (len(iqData) / Config.Fs);

        # Write out data and time.
        asyncDataBuff.write(iqData);
        asyncTimeBuff.write(timeVector);

        # #  Process data if there is enough in the buffers
        if asyncDataBuff.numUnreadSamples() >= sampsForKPulses + overlapSamples:
            print('Buffer Full - sampsForKPulses: %d, overlapSamples: %d,' % (int(sampsForKPulses), int(overlapSamples)))
            print('Running...Buffer full with %d samples. Processing. ' % asyncDataBuff.numUnreadSamples())
            
            tictoc.tic()
            if cleanBuffer:
                # Overlap reads back into the buffer, but there 
                # isn't anything back there on the first segment. 
                # Using an overlap will fill the output with 
                # overlapSamples of zeros at beginning 
                # of x if you specify overlap here. Don't want that
                # so specify zero overlap for first read after
                # cleaning buffer
                x = asyncDataBuff.read(sampsForKPulses);
                t = asyncTimeBuff.read(sampsForKPulses);
                cleanBuffer = False;
            else:
                x = asyncDataBuff.read(sampsForKPulses, overlapSamples);
                t = asyncTimeBuff.read(sampsForKPulses, overlapSamples);

            t0 = t[0];
            print('Running...Building priori and waveform. ')

            # #  PRIMARY PROCESSING BLOCK
            # Prep waveform for processing/detection
            X = Waveform(x, Config.Fs, t0, ps_pre, stftOverlapFraction, Threshold(Config.falseAlarmProb));
            X.K = Config.K;
            print('Current interpulse params || N: %d, M: %d, J: %d' % (int(X.N), int(X.M), int(X.J)))
            X.setPrioriDependentProps(X.ps_pre)
            print('Samples in waveform: %d' % len(X.x))
            tictoc.tic()
            print('Computing STFT...')
            X.spectro();
            print('complete. Elapsed time: %f seconds ' %  tictoc.tocvalue())
            print('Building weighting matrix and generating thresholds...')
            tictoc.tic()
            X.setWeightingMatrix(zetas);

            if suggestedMode == 'S':
                if fLock:
                    mode = 'I';
                else:
                    mode = 'D';
            elif suggestedMode == 'C':
                mode = 'C';
            elif suggestedMode == 'T':
                mode = 'T';
                trackedCount = trackedCount + 1;
            else:
                raise Exception('UAV-RT: Unsupported mode requested. Defaulting to Discovery (D) mode.');

            if Config.opMode == 'freqAllNeverLock':
                mode = 'D';

            if segmentsProcessed == 0:
                X.thresh = X.thresh.makeNewThreshold(X);
            else:
                X.thresh = X.thresh.setThreshold(X, Xhold);

            print('complete. Elapsed time: %f seconds ', tictoc.toc())

            # fprintf('Time windows in S: # u ',int(size(X.stft.S,2)))
            print('Time windows in S: %d' % X.stft.S.shape[1])

            print('Finding pulses...')
            X.process(mode, 'most', Config.excldFreqs)
            processingTime = tictoc.toc();
            print('complete. Elapsed time: %f seconds ', tictoc.toc())

            # #  PREP FOR NEXT LOOP

            # Latch/Release the frequency lock and setup the
            # suggested mode
            suggestedMode = X.ps_pos.mode;
            pulsesConfirmed = np.ndarray.all([X.ps_pos.pl.con_dec]);
            if pulsesConfirmed:
                fLock = True;
            # We only ever release if we are in softlock mode and
            # only do so in that case if we are no longer confirming
            # pulses.
            if Config.opMode == 'freqSearchSoftLock' and not pulsesConfirmed:
                fLock = False;

            # Decide when/how the priori is updated for the next
            # segment's processing.
            if pulsesConfirmed and (mode == 'C' or mode == 'T'):
                X.ps_pos.updatePosteriori(X.ps_pre, X.ps_pos.pl, 'freq')
                if trackedCount > 5:
                    trackedCount = 0;
                    X.ps_pos.updatePosteriori(X.ps_pre, X.ps_pos.pl, 'time')

            # Check lagging processing
            if segmentsProcessed != 0 and Config.K > 1 and processingTime > 0.9 * (sampsForKPulses / Config.Fs):
                Config.K = Config.K - 1;
                print('WARNING!!! PROCESSING TIME TOOK LONGER THAN WAVEFORM LENGTH. STREAMING NOT POSSIBLE. REDUCING NUMBER OF PULSES CONSIDERED BY 1 TO K = %d' % Config.K);
                print('Resetting all internal data buffers and udp buffers to clear potential stale data. ');
                resetBuffersFlag    = True;
                staleDataFlag       = True;
                suggestedMode       = 'S';
            segmentsProcessed = segmentsProcessed + 1;

            tictoc.tic()
            # Prepare priori for next segment
            print('Updating priori...')
            ps_pre = X.ps_pos.deepcopy();

            sampsFprKPulses, overlapSamples = updateBufferReadVariables(X.ps_pos);
            print('complete. Elapsed time: %f seconds ', tictoc.toc())

            Xhold = X.deepcopy()

            for j in range(len(ps_pre.pl)):
                pulse = ps_pre.pl[j]
                print('Pulse at %e Hz detected. SNR: %e Confirmation status: %s' % (pulse.fp, pulse.SNR, pulse.con_dec));
                # udpSenderSend(udpSender, [ pulse.SNR pulse.con_dec pulse.t_0]);

            print('Current Mode: %s' % ps_pre.mode)
            print('====================================')

if __name__ == '__main__':
    uavrt_detection()