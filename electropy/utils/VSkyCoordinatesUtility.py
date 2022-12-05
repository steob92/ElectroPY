import numpy as np
from pyslalib import slalib


def getMJD( i_year, i_month, i_day ):
    return slalib.sla_caldj(i_year,i_month,i_day)[0]


def getUTC( i_mjd, i_seconds ):
    #! and add the fractional day to the mjd to get UTC                                                                                                                                               
    i_utc = i_mjd + i_seconds / 24. / 60. / 60.
    return i_utc


def rotate( theta_rad, x, y):
    s = np.sin( theta_rad );
    c = np.cos( theta_rad );
    _x = x * c - y * s;
    _y = y * c + x * s;
    xx = _x;
    yy = _y;
    return xx, yy


def getWobbleOffset_in_RADec( iNorth, iEast, idec, ira):
    idec  = np.deg2rad(idec);
    ira  = np.deg2rad(ira);

    x = 0.;
    y = 0.;
    z = 1.;
    theta_rad = np.deg2rad(np.sqrt( iNorth * iNorth + iEast * iEast ))
    phi_rad = -1.*np.arctan2( iEast, iNorth );
    if phi_rad < 0:
        phi_rad += 2*np.pi

    z, x = rotate( -theta_rad, z, x );
    y, x = rotate( phi_rad, y, x );
    #// declination                                                                                                                                                                                     
    z, x = rotate( (np.pi/2.) - idec, z, x );
    idiffdec = np.rad2deg((np.arctan2( z, np.sqrt( x * x + y * y ) ) - idec ))
    #// right ascension                                                                                                                                                                                 
    idiffra = np.arctan2( y, x );
    if idiffra < 0.:
        idiffra += 2*np.pi
        
    idiffra *= -1.
    idiffra = np.rad2deg(idiffra)

    if np.abs(idiffra) < 1e-9:
        idiffra = 0.0

    if np.abs(idiffdec) < 1e-9:
        idiffra = 0.0


    return idiffdec, idiffra


def getWobbledDirection( iNorth, iEast, idec, ira):
    dec_W = np.deg2rad(idec)
    ra_W  = np.deg2rad(ira)

    x = 0.;
    y = 0.;
    z = 1.;
    theta_rad = np.deg2rad(np.sqrt( iNorth * iNorth + iEast * iEast ))
    phi_rad = -1.*np.arctan2( iEast, iNorth );
    if phi_rad < 0.:
        phi_rad += 2*np.pi

    z,x = rotate( -theta_rad, z, x );
    y,x = rotate( phi_rad, y, x );
    z,x = rotate( (np.pi/2) - dec_W, z, x );
    x,y = rotate( ra_W, x, y );
    ra_W = np.arctan2( y, x );
    if ra_W < 0.:
        ra_W += 2*np.pi;
    dec_W = np.arctan2( z, np.sqrt( x * x + y * y ) );

    dec_W = np.rad2deg(dec_W)
    ra_W  = np.rad2deg(ra_W)

    return dec_W, ra_W


def precessTarget( iMJD_end, ra, dec, iMJD_start, iUnitIsDeg=True ):
    if iUnitIsDeg:
        ra_rad = np.deg2rad(ra)
        dec_rad = np.deg2rad(dec)

    else:
        ra_rad = ra
        dec_rad = dec

    oy, om, od, ofd, j = slalib.sla_djcl(iMJD_end)
    ny, nd, j= slalib.sla_clyd(oy, om, od)
    ofy_end = ny + nd / 365.25;
    oy, om, od, ofd, j = slalib.sla_djcl(iMJD_start)
    ny, nd, j= slalib.sla_clyd(oy, om, od)
    ofy_start = ny + nd / 365.25;
    ra_rad, dec_rad = slalib.sla_preces('FK5', ofy_start, ofy_end, ra=ra_rad, dc=dec_rad)
    if iUnitIsDeg:
        ra_rad = np.rad2deg(ra_rad)
        dec_rad = np.rad2deg(dec_rad)

    return ra_rad, dec_rad#, ofy_end;


def addToMeanAzimuth( iMean, iAz ):
    if (iMean > 270.) & (iAz < 90.):
        iMean += iAz + 360.
    else:
        iMean += iAz;

    return iMean;


def getTargetShiftWest( iTargetRA_deg, iTargetDec_deg, ira_deg, idec_deg ):
    sep  = slalib.sla_dsep(np.deg2rad(iTargetRA_deg), np.deg2rad(iTargetDec_deg), np.deg2rad(ira_deg), np.deg2rad(idec_deg));
    bear = slalib.sla_bear(np.deg2rad(iTargetRA_deg), np.deg2rad(iTargetDec_deg), np.deg2rad(ira_deg), np.deg2rad(idec_deg));

    iShift = np.rad2deg(sep * np.sin(bear))
    if np.abs( iShift ) < 1.e-8: 
        iShift = 0.;

    return iShift;


def getTargetShiftNorth( iTargetRA_deg, iTargetDec_deg, ira_deg, idec_deg ):
    sep  = slalib.sla_dsep(np.deg2rad(iTargetRA_deg), np.deg2rad(iTargetDec_deg), np.deg2rad(ira_deg), np.deg2rad(idec_deg));
    bear = slalib.sla_bear(np.deg2rad(iTargetRA_deg), np.deg2rad(iTargetDec_deg), np.deg2rad(ira_deg), np.deg2rad(idec_deg));

    iShift = np.rad2deg(sep * np.cos(bear))
    if np.abs( iShift ) < 1.e-8:
        iShift = 0.
	
    return iShift;


def convert_derotatedCoordinates_to_J2000( iMJD, i_RA_J2000_deg, i_DEC_J2000_deg, x, y):
    i_ra = np.deg2rad(i_RA_J2000_deg)
    i_dec = np.deg2rad(i_DEC_J2000_deg)


 #   print("Original i_ra={}, i_dec={}".format(i_ra, i_dec))
    i_ra, i_dec = precessTarget( iMJD, i_RA_J2000_deg, i_DEC_J2000_deg, 51544, True );
  #  print("Precessed i_ra={}, i_dec={}".format(i_ra, i_dec))

    # // calculate wobble offset in ra/dec for current epoch                                                                                                                                             
    i_decDiff, i_raDiff = getWobbleOffset_in_RADec( y, -x, i_dec, i_ra);

#    print("i_decDiff ={}, i_radiff={}".format(i_decDiff, i_raDiff))
    if i_raDiff < -180.:
        i_raDiff += 360.;

    i_decWobble = i_dec + i_decDiff;
    i_raWobble  = i_ra  + i_raDiff;

#    print("i_decWobble ={}, i_raWobble={}".format(i_decWobble, i_raWobble))

    #        // correct for precession (from current epoch to J2000=MJD51544)                                                                                                                                   
    i_raWobble, i_decWobble = precessTarget( 51544.,i_raWobble, i_decWobble, iMJD, True );

   # print("i_decWobble ={}, i_raWobble={}".format(i_decWobble, i_raWobble))
    x = getTargetShiftWest( i_RA_J2000_deg, i_DEC_J2000_deg, i_raWobble, i_decWobble ) * -1.;
    y = getTargetShiftNorth( i_RA_J2000_deg, i_DEC_J2000_deg, i_raWobble, i_decWobble );
    return x, y


def getHorizontalCoordinates( MJD, time, dec_deg, ra_deg):
    Observatory_Latitude = 31.675 
    Observatory_Longitude = 110.952 
    
    # // convert time to fraction of a day                                                                                                                                                               
    iTime = time / 86400.;
    # // get Greenwich sideral time                                                                                                                                                                      
    iSid = slalib.sla_gmsta(MJD, iTime);
    #// calculate local sideral time                                                                                                                                                                    
    iSid = iSid - np.deg2rad(Observatory_Longitude);
    #// calculate hour angle                                                                                                                                                                            
    ha = slalib.sla_dranrm( iSid - np.deg2rad(ra_deg));
    #// get horizontal coordinates                                                                                                                                                                      
    az_deg, ele_deg = slalib.sla_de2h( ha, np.deg2rad(dec_deg), np.deg2rad(Observatory_Latitude));
    #// from [rad] to [deg]                                                                                                                                                                             
    ele_deg = np.rad2deg(ele_deg)
    az_deg = np.rad2deg(az_deg)

    return ele_deg, az_deg




 


