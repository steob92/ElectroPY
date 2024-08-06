import numpy as np
from pyslalib import slalib
from typing import Union, Tuple, Optional


def get_mjd( 
        i_year : Union[int,float], 
        i_month : Union[int,float], 
        i_day : Union[int,float] 
    ) -> float:
    """Get the MJD for a given year/month/day

    Parameters
    ----------
    i_year : Union[int,float]
        year of the date to be converted
    i_month : Union[int,float]
        month of the date to be converted
    i_day : Union[int,float]
        day of the date to be converted

    Returns
    -------
    float
        date in MJD
        
    
    """
    return slalib.sla_caldj(i_year,i_month,i_day)[0]


def get_utc( 
        i_mjd : Union[int,float], 
        i_seconds : Union[int,float] 
    ) -> float:
    """Get the fractional UTC date

    Parameters
    ----------
    i_mjd : Union[int,float]
        mjd of the date to be converted
    i_seconds : Union[int,float]
        number of seconds

    Returns
    -------
    float
        UTC date expressed as a float

    """
    i_utc = i_mjd + i_seconds / 24. / 60. / 60.
    return i_utc


def rotate( 
        theta_rad : float, 
        x : Union[float, np.ndarray], 
        y : Union[float, np.ndarray]
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Rotate coordinates about an angle

    Parameters:
    theta_rad : float
        angle to rotate about in radians
    x : Union[float, np.ndarray]
        x-coordinate to be rotated
    y : Union[float, np.ndarray]
        y-coordinate to be rotated
    
        
    Returns:
    Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]
        rotated x and y coordintates
    
    """
    c = np.cos( theta_rad )
    s = np.sin( theta_rad )
    _x = x * c - y * s
    _y = y * c + x * s
    xx = _x
    yy = _y
    return xx, yy


def get_wobble_offset_in_radec( 
        i_north : float,
        i_east : float,
        i_dec : float,
        i_ra : float 
    ) -> Tuple[float, float]:
    """Convert the wobble offset in North/East to Ra/Dec
    
    Parameters
    ----------
    i_north : float
        north wobble angle
    i_east : float
        east wobble angle
    i_dec : float
        declination angle of the centre of the FoV in degrees
    i_ra : float
        right ascension angle of the centre of the FoV in degrees
    
        
    Returns
    -------
    Tuple[float, float]
        declination and right ascension offset of the wobble
    """

    i_dec  = np.deg2rad(i_dec)
    i_ra  = np.deg2rad(i_ra)

    x = 0.
    y = 0.
    z = 1.
    theta_rad = np.deg2rad(np.sqrt( i_north * i_north + i_east * i_east ))
    phi_rad = -1.*np.arctan2( i_east, i_north )
    if phi_rad < 0:
        phi_rad += 2*np.pi

    z, x = rotate( -theta_rad, z, x )
    y, x = rotate( phi_rad, y, x )
    #  declination                                                                                                                                                                                     
    z, x = rotate( (np.pi/2.) - i_dec, z, x )
    idiffdec = np.rad2deg((np.arctan2( z, np.sqrt( x * x + y * y ) ) - i_dec ))
    #  right ascension                                                                                                                                                                                 
    idiffra = np.arctan2( y, x )
    if idiffra < 0.:
        idiffra += 2*np.pi
        
    idiffra *= -1.
    idiffra = np.rad2deg(idiffra)

    if np.abs(idiffra) < 1e-9:
        idiffra = 0.0

    if np.abs(idiffdec) < 1e-9:
        idiffra = 0.0


    return idiffdec, idiffra


def get_wobbled_direction( 
        i_north : float, 
        i_east : float, 
        i_dec : float, 
        i_ra : float
    ) -> Tuple[float, float]:
    """Get the wobble direct 
    ToDO: Get more details

    Parameters
    ----------
    i_north : float
        northern wobble
    i_east : float
        eastern wobble
    i_dec : float
        declination angle in degrees
    i_ra : float
        right ascension in degrees

    Returns
    -------
    Tuple[float, float]
        shifted declination and right ascension in degrees

    """
    dec_w = np.deg2rad(i_dec)
    ra_w  = np.deg2rad(i_ra)

    x = 0.
    y = 0.
    z = 1.
    theta_rad = np.deg2rad(np.sqrt( i_north * i_north + i_east * i_east ))
    phi_rad = -1.*np.arctan2( i_east, i_north )
    if phi_rad < 0.:
        phi_rad += 2*np.pi

    z,x = rotate( -theta_rad, z, x )
    y,x = rotate( phi_rad, y, x )
    z,x = rotate( (np.pi/2) - dec_w, z, x )
    x,y = rotate( ra_w, x, y )
    ra_w = np.arctan2( y, x )
    if ra_w < 0.:
        ra_w += 2*np.pi
    dec_w = np.arctan2( z, np.sqrt( x * x + y * y ) )

    dec_w = np.rad2deg(dec_w)
    ra_w  = np.rad2deg(ra_w)

    return dec_w, ra_w


def precess_target( 
        i_mjd_end : float,
        ra : float,
        dec : float,
        i_mjd_start : float,
        is_degrees : Optional[bool] = True 
    ) -> Tuple[float, float]:
    """Precess target based on sky location and dates

    Parameters
    ----------
    i_mjd_end : float
        MJD of the end date
    ra : float
        target right ascension
    dec : float
        target declination
    i_mjd_start : float
        MJD of the start date
    is_degrees : Optional[bool]
        whether or not the ra/dec are in degrees. Default is True

    Returns
    -------
    Tuple[float, float]
        right ascension and declination of the target in radians at the end date
    """
    if is_degrees:
        ra_rad = np.deg2rad(ra)
        dec_rad = np.deg2rad(dec)

    else:
        ra_rad = ra
        dec_rad = dec

    oy, om, od, ofd, j = slalib.sla_djcl(i_mjd_end)
    ny, nd, j= slalib.sla_clyd(oy, om, od)
    ofy_end = ny + nd / 365.25
    oy, om, od, ofd, j = slalib.sla_djcl(i_mjd_start)
    ny, nd, j= slalib.sla_clyd(oy, om, od)
    ofy_start = ny + nd / 365.25
    ra_rad, dec_rad = slalib.sla_preces('FK5', ofy_start, ofy_end, ra=ra_rad, dc=dec_rad)
    if is_degrees:
        ra_rad = np.rad2deg(ra_rad)
        dec_rad = np.rad2deg(dec_rad)

    return ra_rad, dec_rad


def add_to_mean_azimuth( i_mean : float, i_az : float ) -> float:
    """ Add a value to a mean azimuth

    Correct for cicular coordinates
    
    Parameters
    ----------
    i_mean : float
        mean value that we're adding to
    i_az : float
        value to be added

    Returns
    -------
    float
        new mean value
    """
    if (i_mean > 270.) & (i_az < 90.):
        i_mean += i_az + 360.
    else:
        i_mean += i_az

    return i_mean


def get_target_shift_west( 
        i_target_ra_deg : float, 
        i_target_dec_deg : float, 
        i_ra_deg : float, 
        i_dec_deg : float
    ) -> float:
    """Get the shift in the western shift with respect to a target

    Parameters
    ----------
    i_target_ra_deg : float
        right ascension of the target in degrees
    i_target_dec_deg : float
        declination of the target in degrees
    i_ra_deg : float
        right ascension of the sky location in degrees
    i_dec_deg : float
        declination of the sky location in degrees

    Returns
    -------
    float
        Western shift in degrees
    """
    sep  = slalib.sla_dsep(np.deg2rad(i_target_ra_deg), np.deg2rad(i_target_dec_deg), np.deg2rad(i_ra_deg), np.deg2rad(i_dec_deg))
    bear = slalib.sla_bear(np.deg2rad(i_target_ra_deg), np.deg2rad(i_target_dec_deg), np.deg2rad(i_ra_deg), np.deg2rad(i_dec_deg))

    iShift = np.rad2deg(sep * np.sin(bear))
    if np.abs( iShift ) < 1.e-8: 
        iShift = 0.

    return iShift


def get_target_shift_north( 
        i_target_ra_deg : float, 
        i_target_dec_deg : float, 
        i_ra_deg : float, 
        i_dec_deg : float
    ) -> float:
    """Get the shift in the northern shift with respect to a target

    Parameters
    ----------
    i_target_ra_deg : float
        right ascension of the target in degrees
    i_target_dec_deg : float
        declination of the target in degrees
    i_ra_deg : float
        right ascension of the sky location in degrees
    i_dec_deg : float
        declination of the sky location in degrees

    Returns
    -------
    float
        Northern shift in degrees
    """
    sep  = slalib.sla_dsep(
        np.deg2rad(i_target_ra_deg), 
        np.deg2rad(i_target_dec_deg), 
        np.deg2rad(i_ra_deg), 
        np.deg2rad(i_dec_deg)
    )
    bear = slalib.sla_bear(
        np.deg2rad(i_target_ra_deg), 
        np.deg2rad(i_target_dec_deg), 
        np.deg2rad(i_ra_deg), 
        np.deg2rad(i_dec_deg)
    )

    iShift = np.rad2deg(sep * np.cos(bear))
    if np.abs( iShift ) < 1.e-8:
        iShift = 0.
	
    return iShift


def convert_derotated_coordinates_to_J2000( 
        i_mjd : float, 
        i_ra_J2000_deg : float, 
        i_dec_J2000_deg : float, 
        x : float, 
        y : float
    ) -> Tuple[float, float]:
    """ Derotate the coordintaes

    Parameters
    ----------
    i_mjd : float
        MJD of the observations
    i_ra_J2000_deg : float
        Right acension in J2000 and degrees
    i_dec_J2000_deg : float
        Declination in J2000 and degrees
    x : float
        x offset
    y : float
        y offset

    Returns
    -------
    Tuple[float, float]
        Rerotated x,y coordinates
    """
    i_ra = np.deg2rad(i_ra_J2000_deg)
    i_dec = np.deg2rad(i_dec_J2000_deg)


    #   print("Original i_ra={}, i_dec={}".format(i_ra, i_dec))
    i_ra, i_dec = precess_target( i_mjd, i_ra_J2000_deg, i_dec_J2000_deg, 51544, True )
    #  print("Precessed i_ra={}, i_dec={}".format(i_ra, i_dec))

    # // calculate wobble offset in ra/dec for current epoch                                                                                                                                             
    i_decDiff, i_raDiff = get_wobble_offset_in_radec( y, -x, i_dec, i_ra)

    #    print("i_decDiff ={}, i_radiff={}".format(i_decDiff, i_raDiff))
    if i_raDiff < -180.:
        i_raDiff += 360.

    i_decWobble = i_dec + i_decDiff
    i_raWobble  = i_ra  + i_raDiff

    #    print("i_decWobble ={}, i_raWobble={}".format(i_decWobble, i_raWobble))
    #        // correct for precession (from current epoch to J2000=MJD51544)                                                                                                                                   
    i_raWobble, i_decWobble = precess_target( 51544.,i_raWobble, i_decWobble, i_mjd, True )

    # print("i_decWobble ={}, i_raWobble={}".format(i_decWobble, i_raWobble))
    x = get_target_shift_west( i_ra_J2000_deg, i_dec_J2000_deg, i_raWobble, i_decWobble ) * -1.
    y = get_target_shift_north( i_ra_J2000_deg, i_dec_J2000_deg, i_raWobble, i_decWobble )
    return x, y


def get_horizontal_coordinates(
        mjd : float,
        time : float,
        dec_deg : float,
        ra_deg : float
    ) -> Tuple[float, float]:
    """Get the horizontal coordinates (elevation, azimuth) for a given RA/Dec at a given time
    
    Parameters
    ----------
    mjd : float
        MJD of observation
    time : float
        integer seconds of the dat of the observation
    dec_deg : float
        declination of the target location in degrees
    ra_deg : float
        right acension of the target location in degrees

    Returns
    -------
    Tuple[float, float]
        elevation and azimuth of the location in degrees
    """
    observatory_latitude = 31.675 
    observatory_longitude = 110.952 
    
    # // convert time to fraction of a day                                                                                                                                                               
    i_time = time / 86400.
    # // get Greenwich sideral time                                                                                                                                                                      
    i_sid = slalib.sla_gmsta(mjd, i_time)
    #  calculate local sideral time                                                                                                                                                                    
    i_sid = i_sid - np.deg2rad(observatory_longitude)
    #  calculate hour angle                                                                                                                                                                            
    ha = slalib.sla_dranrm( i_sid - np.deg2rad(ra_deg))
    #  get horizontal coordinates                                                                                                                                                                      
    az_deg, ele_deg = slalib.sla_de2h( ha, np.deg2rad(dec_deg), np.deg2rad(observatory_latitude))
    #  from [rad] to [deg]                                                                                                                                                                             
    ele_deg = np.rad2deg(ele_deg)
    az_deg = np.rad2deg(az_deg)

    return ele_deg, az_deg




 


