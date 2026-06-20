"""Hardcoded mapping of PGA Tour event names to course names."""
# fmt: off
EVENT_COURSE_MAP = {
    # Sentry Tournament of Champions / The Sentry
    "Sentry Tournament of Champions": "Kapalua Resort (Plantation Course)",
    "The Sentry": "Kapalua Resort (Plantation Course)",
    
    # Sony Open in Hawaii
    "Sony Open in Hawaii": "Waialae Country Club",
    
    # The American Express / CareerBuilder Challenge
    "The American Express": "PGA West (Pete Dye Stadium Course)",
    "CareerBuilder Challenge": "PGA West (Pete Dye Stadium Course)",
    
    # Farmers Insurance Open
    "Farmers Insurance Open": "Torrey Pines Golf Course (South Course)",
    
    # AT&T Pebble Beach Pro-Am
    "AT&T Pebble Beach Pro-Am": "Pebble Beach Golf Links",
    
    # WM Phoenix Open / Waste Management Phoenix Open
    "WM Phoenix Open": "TPC Scottsdale (Stadium Course)",
    "Waste Management Phoenix Open": "TPC Scottsdale (Stadium Course)",
    
    # Genesis Invitational / Genesis Open / Northern Trust
    "Genesis Invitational": "Riviera Country Club",
    "Genesis Open": "Riviera Country Club",
    
    # Mexico Open at Vidanta
    "Mexico Open at Vidanta": "Vidanta Vallarta",
    "WGC-Mexico Championship": "Club de Golf Chapultepec",
    
    # Cognizant Classic / The Honda Classic
    "Cognizant Classic": "PGA National Resort (The Champion Course)",
    "The Honda Classic": "PGA National Resort (The Champion Course)",
    
    # Arnold Palmer Invitational
    "Arnold Palmer Invitational pres. by Mastercard": "Arnold Palmer's Bay Hill Club & Lodge",
    "Arnold Palmer Invitational": "Arnold Palmer's Bay Hill Club & Lodge",
    
    # THE PLAYERS Championship
    "THE PLAYERS Championship": "TPC Sawgrass (THE PLAYERS Stadium Course)",
    
    # Valspar Championship
    "Valspar Championship": "Innisbrook Resort (Copperhead Course)",
    
    # Texas Children's Houston Open / Houston Open
    "Texas Children's Houston Open": "Memorial Park Golf Course",
    "Houston Open": "Memorial Park Golf Course",
    
    # Valero Texas Open
    "Valero Texas Open": "TPC San Antonio (Oaks Course)",
    
    # Masters Tournament
    "Masters Tournament": "Augusta National Golf Club",
    
    # RBC Heritage
    "RBC Heritage": "Harbour Town Golf Links",
    
    # Zurich Classic of New Orleans
    "Zurich Classic of New Orleans": "TPC Craig Ranch",
    
    # Wells Fargo Championship
    "Wells Fargo Championship": "Quail Hollow Club",
    
    # AT&T Byron Nelson
    "AT&T Byron Nelson": "TPC Craig Ranch",
    "BYRON NELSON": "TPC Craig Ranch",
    
    # Charles Schwab Challenge / Dean & DeLuca Invitational
    "Charles Schwab Challenge": "Colonial Country Club",
    "Dean & DeLuca Invitational": "Colonial Country Club",
    
    # RBC Canadian Open
    "RBC Canadian Open": "Hamilton Golf & Country Club",
    "RBC Canadian Open presented by Rolex": "Hamilton Golf & Country Club",
    
    # The Memorial Tournament
    "The Memorial Tournament presented by Workday": "Muirfield Village Golf Club",
    "The Memorial Tournament": "Muirfield Village Golf Club",
    
    # Travelers Championship
    "Travelers Championship": "TPC River Highlands",
    
    # Rocket Mortgage Classic
    "Rocket Mortgage Classic": "Detroit Golf Club",
    
    # John Deere Classic
    "John Deere Classic": "TPC Deere Run",
    
    # Genesis Scottish Open
    "Genesis Scottish Open": "The Renaissance Club",
    
    # The Open Championship
    "The Open Championship": "Royal Troon Golf Club",
    "The Open": "Royal Troon Golf Club",
    
    # 3M Open
    "3M Open": "TPC Twin Cities",
    
    # Wyndham Championship
    "Wyndham Championship": "Sedgefield Country Club",
    
    # FedEx St. Jude Championship
    "FedEx St. Jude Championship": "TPC Southwind",
    "World Golf Championships-FedEx St. Jude Invitational": "TPC Southwind",
    
    # BMW Championship
    "BMW Championship": "Olympia Fields Country Club",
    
    # TOUR Championship
    "TOUR Championship": "East Lake Golf Club",
    
    # Procore Championship / Fortinet Championship
    "Procore Championship": "Silverado Resort (North Course)",
    "Fortinet Championship": "Silverado Resort (North Course)",
    
    # Sanderson Farms Championship
    "Sanderson Farms Championship": "The Country Club of Jackson",
    
    # Shriners Children's Open
    "Shriners Children's Open": "TPC Summerlin",
    
    # ZOZO CHAMPIONSHIP
    "ZOZO CHAMPIONSHIP": "Accordia Golf Narashino Country Club",
    
    # World Wide Technology Championship
    "World Wide Technology Championship": "El Cardonal at Diamante",
    "Mayakoba Golf Classic": "El Camaleon Golf Club",
    
    # Butterfield Bermuda Championship
    "Butterfield Bermuda Championship": "Port Royal Golf Course",
    
    # The RSM Classic
    "The RSM Classic": "Sea Island Golf Club (Seaside Course)",
    
    # Puerto Rico Open
    "Puerto Rico Open": "Grand Reserve Golf Club",
    
    # Corales Puntacana Championship
    "Corales Puntacana Championship": "Puntacana Resort & Club (Corales Golf Course)",
    "Corales Puntacana Resort & Club Championship": "Puntacana Resort & Club (Corales Golf Course)",
    
    # Barracuda Championship
    "Barracuda Championship": "Tahoe Mountain Club (Old Greenwood)",
    
    # Myrtle Beach Classic
    "Myrtle Beach Classic": "Dunes Golf and Beach Club",
    
    # Black Desert Championship
    "Black Desert Championship": "Black Desert Resort Golf Course",
    
    # ISCO Championship
    "ISCO Championship": "Keene Trace Golf Club (Champions Course)",
    
    # Palmetto Championship
    "Palmetto Championship at Congaree": "Congaree Golf Club",
    
    # Palmetto Championship
    "RBC Heritage presented by BOSE": "Harbour Town Golf Links",
    
    # WGC-Dell Technologies Match Play
    "WGC-Dell Technologies Match Play": "Austin Country Club",
    "WGC-Match Play": "Austin Country Club",
    
    # WGC-FedEx St. Jude Invitational
    "World Golf Championships-FedEx St. Jude Invitational": "TPC Southwind",
    
    # WGC-Workday Championship
    "WGC-Workday Championship": "The Concession Golf Club",
    "WGC-Mexico Championship": "Club de Golf Chapultepec",
    
    # The Northern Trust
    "The Northern Trust": "Liberty National Golf Club",
    
    # FedEx Cup Playoffs
    "The Northern Trust presented by Skyfall": "Liberty National Golf Club",
    
    # Sentry Tournament of Champions variants
    "Sentry Tournament of Champions": "Kapalua Resort (Plantation Course)",
    
    # Utah Championship
    "Utah Championship": "Oakridge Country Club",
    
    # AdventHealth Championship
    "AdventHealth Championship": "Blue Hills Country Club",
    
    # Simmons Bank Open
    "Simmons Bank Open for the Snedeker Foundation": "The Grove Golf Club",
    "Simmons Bank Open": "The Grove Golf Club",
    
    # Korn Ferry Tour Finals
    "Korn Ferry Tour Championship": "Fallasburg Park Golf Course",
    
    # Other WGC events
    "WGC-HSBC Champions": "Sheshan International Golf Club",
    
    # Match Play
    "WGC-Accenture Match Play Championship": "The Ritz-Carlton Golf Club",
    
    # Year-prefixed events
    "2018 Masters Tournament": "Augusta National Golf Club",
    "2019 Masters Tournament": "Augusta National Golf Club",
    "2020 Masters Tournament": "Augusta National Golf Club",
    "2021 Masters Tournament": "Augusta National Golf Club",
    
    # Events with slight name variations
    "Arnold Palmer Invitational presented by Mastercard": "Arnold Palmer's Bay Hill Club & Lodge",
    "The Memorial Tournament pres. by Workday": "Muirfield Village Golf Club",
    "the Memorial Tournament pres. by Workday": "Muirfield Village Golf Club",
    "the Memorial Tournament pres. by Nationwide": "Muirfield Village Golf Club",
    "The Genesis Invitational": "Riviera Country Club",
    "Cadence Bank Houston Open": "Memorial Park Golf Course",
    "Cognizant Classic in The Palm Beaches": "PGA National Resort (The Champion Course)",
    "Vivint Houston Open": "Memorial Park Golf Course",
    "Bermuda Championship": "Port Royal Golf Course",
    "Rocket Classic": "Detroit Golf Club",
    "Mexico Open": "Vidanta Vallarta",
    "Mexico Open at VidantaWorld": "Vidanta Vallarta",
    "ONEflight Myrtle Beach Classic": "Dunes Golf and Beach Club",
    "THE CJ CUP Byron Nelson": "TPC Craig Ranch",
    "THE CJ CUP in South Carolina": "Congaree Golf Club",
    "THE CJ CUP @ NINE BRIDGES": "Nine Bridges Golf Club",
    "THE CJ CUP @ SHADOW CREEK": "Shadow Creek Golf Course",
    "THE CJ CUP @ SUMMIT": "The Summit Club",
    "Truist Championship": "Quail Hollow Club",
    "The ZOZO CHAMPIONSHIP": "Accordia Golf Narashino Country Club",
    "World Wide Technology Championship at Mayakoba": "El Camaleon Golf Club",
    "Mayakoba Golf Classic presented by UNIFIN": "El Camaleon Golf Club",
    "WGC-Workday Championship at The Concession": "The Concession Golf Club",
    "WGC-FedEx St. Jude Invitational": "TPC Southwind",
    "FedEx St. Jude Classic": "TPC Southwind",
    "World Golf Championships-Mexico Championship": "Club de Golf Chapultepec",
    "Workday Charity Open": "Muirfield Village Golf Club",
    "Desert Classic": "PGA West (La Quinta Country Club)",
    "A Military Tribute at The Greenbrier": "The Old White TPC",
    "Safeway Open": "Silverado Resort (North Course)",
    "Quicken Loans National": "TPC Potomac at Avenel Farm",
    "Barbasol Championship": "Keene Trace Golf Club (Champions Course)",
    "Bank of Utah Championship": "Woods Cross Golf Club",
    "FedEx Cup Playoffs - BMW Championship": "Olympia Fields Country Club",
    
    # Majors
    "PGA Championship": "Valhalla Golf Club",
    "U.S. Open": "Pinehurst Resort (Course No. 2)",
    "The Open Championship": "Royal Troon Golf Club",
    "Ryder Cup": "Bethpage Black",
    "Presidents Cup": "Royal Montreal Golf Club",
    
    # Special events
    "Hero World Challenge": "Albany Golf Club",
    "The Match": "Shadow Creek Golf Course",
    "The Match: Champions for Change": "Stone Canyon Golf Club",
    "The Match: Champions for Charity": "Medalist Golf Club",
    "Crypto.com Showdown": "Shadow Creek Golf Course",
    
    # Shriners variants
    "Shriners Hospital for Children Open": "TPC Summerlin",
    "Shriners Hospitals for Children Open": "TPC Summerlin",
    
    # CJ CUP
    "THE CJ CUP": "Nine Bridges Golf Club",
    
    # Name variations for known events
    "Baycurrent Classic": "Capilano Golf & Country Club",
    
    # Q-School
    "PGA TOUR Q-School presented by Korn Ferry": "TPC Sawgrass (Dye's Valley Course)",
    
    # Fort Worth Invitational
    "Fort Worth Invitational": "Colonial Country Club",
    
    # CIMB Classic
    "CIMB Classic": "Kuala Lumpur Golf and Country Club",
    
    # Last 2
    "Dell Technologies Championship": "TPC Boston",
    "WGC-Bridgestone Invitational": "Firestone Country Club (South Course)",
}
# fmt: on


def get_course_name(event_name):
    """Look up course name for an event. Returns None if not found."""
    # Direct match
    if event_name in EVENT_COURSE_MAP:
        return EVENT_COURSE_MAP[event_name]
    
    # Case-insensitive search
    name_lower = event_name.lower().strip()
    for key, val in EVENT_COURSE_MAP.items():
        if key.lower().strip() == name_lower:
            return val
    
    return None
