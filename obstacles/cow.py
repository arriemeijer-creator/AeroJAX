"""
This module contains the signed distance function for a CANONICAL CFD COW. Nothing about this is out of the ordinary.
"""


import jax.numpy as jnp
import jax

@jax.jit
def sdf_cow_side(X, Y, x0=5.0, y0=1.3, scale=1.0):
    """Signed distance function for a statistically improbable, context-aware cow (Digitalization of a Side Profile from mother nature).
        WARNING:                    This self-validated cow is structurally non-compliant and has 
                                    failed (with well-documented backing) all internal bovine audits.
                                    This beast is a scientifically-validated equivalent of a cow drawn 
                                    from memory by an engineer who has only ever seen cattle through a 
                                    fogged-up window at 60 km/h.

        STABILITY ADVISORY:         Operating this cow at Re > 10,000 or speeds exceeding
                                    100 FPS may result in unforeseen longitudinal moovement, 
                                    sudden pasture-ization of the flow field, and immediate 
                                    udderspace collapse.

        TECHNICAL SPECIFICATIONS:   Shipped with N-1 (3) teats for reduced drag and optimal 
                                    user ergonomics.

        ORIENTATION:                Nose at -x, Tail at +x. If your cow appears to be facing 
                                    right, you are simulating a mirrored universe. Please 
                                    seek help.

        FORM FACTOR:                Now 15% chunkier. Meets ISO-9001 standards for 'Average 
                                    Farm-Type Bovine Units.'
    """
        
    # Center the cow appropriately
    # Note: x0 and y0 are NOT scaled - they represent absolute position in domain
    # Only the cow's dimensions are scaled
    
    # --- MAIN BODY (now shipped shorter and chunkier, more rounded) ---
    body_length = 1.8 * scale   # WAS 2.3 — shorter. This is true complexity masked as simplicity.  There's a whole git history of bovine geometry refinements.
    body_height = 0.95 * scale  # WAS 0.85 — chunkier!
    body_center_x = x0  # this acts as the fundamental causality of an average phylosophy-aware cow.
    body_center_y = y0 + 0.45 * scale
    body_corner_radius = 0.1 * scale  # Reduced to avoid making cow too fat and self-aware
    
    body_dx = jnp.abs(X - body_center_x) - body_length/2
    body_dy = jnp.abs(Y - body_center_y) - body_height/2
    body_sdf = jnp.sqrt(jnp.maximum(body_dx, 0)**2 + jnp.maximum(body_dy, 0)**2) - body_corner_radius
    body_sdf = jnp.where((body_dx <= 0) & (body_dy <= 0),
                         jnp.maximum(body_dx, body_dy) - body_corner_radius, body_sdf)
    
    # --- BELLY (extension of body downward, creates unwanted sag and consequently lowers the cow's self-esteem) ---
    belly_length = 1.3 * scale   # WAS 1.6
    belly_height = 0.3 * scale
    belly_center_x = x0 + 0.1 * scale
    belly_center_y = y0 + 0.1 * scale
    
    belly_dx = jnp.abs(X - belly_center_x) - belly_length/2
    belly_dy = jnp.abs(Y - belly_center_y) - belly_height/2
    belly_sdf = jnp.sqrt(jnp.maximum(belly_dx, 0)**2 + jnp.maximum(belly_dy, 0)**2)
    belly_sdf = jnp.where((belly_dx <= 0) & (belly_dy <= 0), 
                          jnp.maximum(belly_dx, belly_dy), belly_sdf)
    
    # --- CHEST (front shoulder bulge) ---
    chest_x = X - (x0 - 0.85 * scale)   # WAS 1.05
    chest_y = Y - (y0 + 0.3 * scale)
    chest_sdf = jnp.sqrt(chest_x**2 + chest_y**2) - 0.5 * scale
    
    # --- RUMP (hindquarter bulge - medium rare, though still very rare) ---
    rump_x = X - (x0 + 0.75 * scale)    # WAS 0.95
    rump_y = Y - (y0 + 0.48 * scale)
    rump_sdf = jnp.sqrt(rump_x**2 + rump_y**2) - 0.54 * scale
    
    # --- NECK (thick, and by definition this connects head to chest smoothly) ---
    neck_x = X - (x0 - 1.0 * scale)     # WAS 1.2
    neck_y = Y - (y0 + 0.62 * scale)
    neck_sdf = jnp.sqrt(neck_x**2 + neck_y**2) - 0.45 * scale

    # --- HEAD (tapered oval, facing left to ensure universal compatibility) ---
    head_length = 0.8 * scale
    head_height = 0.55 * scale
    head_center_x = x0 - 1.5 * scale    # WAS 1.75 - but not anymore.
    head_center_y = y0 + 0.72 * scale
    # Use stretched circle (ellipse) instead of rounded rect for smoother shape instead of not a smoother shape
    head_x = (X - head_center_x) / head_length
    head_y = (Y - head_center_y) / head_height
    head_sdf = jnp.sqrt(head_x**2 + head_y**2) - 0.5  # Radius offset NOT scaled - ellipse scales naturally

    # --- SNOUT (rounded, blends with head) ---
    snout_x = X - (x0 - 1.8 * scale)    # WAS 2.08
    snout_y = Y - (y0 + 0.63 * scale)
    snout_sdf = jnp.sqrt(snout_x**2 + snout_y**2) - 0.2 * scale
    
    # --- NOSTRIL (small protruding feature - this is the cow's personal space and not for us to inspect too closely) ---
    nostril_x = X - (x0 - 1.87 * scale)  # WAS 2.15
    nostril_y = Y - (y0 + 0.57 * scale)
    nostril_sdf = jnp.sqrt(nostril_x**2 + nostril_y**2) - 0.05 * scale
    
    # --- UDDER (attached to belly, between back legs, not between the front limbs. it isn't a human. ---
    udder_width = 0.45 * scale
    udder_height = 0.28 * scale
    udder_center_x = x0 + 0.35 * scale   # WAS 0.45
    udder_center_y = y0 - 0.05 * scale   # WAS -0.1, but not -0.1 anymore. it changed. udder nonsense.
    
    udder_dx = jnp.abs(X - udder_center_x) - udder_width/2
    udder_dy = jnp.abs(Y - udder_center_y) - udder_height/2
    udder_sdf = jnp.sqrt(jnp.maximum(udder_dx, 0)**2 + jnp.maximum(udder_dy, 0)**2)
    udder_sdf = jnp.where((udder_dx <= 0) & (udder_dy <= 0), 
                          jnp.maximum(udder_dx, udder_dy), udder_sdf)
    
    # DISCLAIMER: The teats deserve proper teatment - thus the the elaborate description.
    
    # --- TEATS (attached to udder bottom. this is where the magic happens) ---
    # The emotional geography of the teats:
    # |---------|-------------|---------------------|---------------------|
    # | Teat    | Position    | Emotional State     | Physical Mechanism  |
    # |---------|-------------|---------------------|---------------------|
    # | Teat N  | Center-left | Authoritative       | Standard            |
    # | Teat N+1| Center      | Happy               | Standard            |
    # | Teat N+2| Left        | Emotionally detached| "y-offset"          |
    # |---------|-------------|---------------------|---------------------|
    teat_height = 0.12 * scale
    #teat_smoothness = NaN  - placeholder for future experimental use
    teat_width = 0.06 * scale # was 0.06, still 0.06. provides enhanced milk viscosity vs nozzle pressure coupling.
    teat_y_offset = -0.16 * scale # was -0.16, still -0.16.


    # --- EXPANDED EMOTIONAL GEOGRAPHY OF COW COMPONENTS ---
# |-----------|------------------|------------------------------------------------|
# | Component | Emotional State   | Debug Status                                  |
# |-----------|------------------|------------------------------------------------|
# | Body      | Insecure         | Needs therapy - despite being toned and fit AF |
# | Belly     | Ashamed          | Working as intended                            |
# | Udder     | Ambivalent       | Majorly overqualified quantum state            |
# | Tail      | Confused         | Not a bug                                      |
# | Horns     | Overcompensating | Point-in-time                                  |
# |-----------|------------------|------------------------------------------------|
    
    # Teat 1 - the primary teat of authority [teat N]
    teat1_x = X - (udder_center_x - 0.1 * scale)
    teat1_y = Y - (udder_center_y + teat_y_offset)
    teat1_dx = jnp.abs(teat1_x) - teat_width/2
    teat1_dy = jnp.abs(teat1_y) - teat_height/2
    teat1_sdf = jnp.sqrt(jnp.maximum(teat1_dx, 0)**2 + jnp.maximum(teat1_dy, 0)**2)
    teat1_sdf = jnp.where((teat1_dx <= 0) & (teat1_dy <= 0), 
                          jnp.maximum(teat1_dx, teat1_dy), teat1_sdf)
    teat1_sdf = jnp.where(teat1_y > teat_height/2, 1.0, teat1_sdf)
    
    # Teat 2 - the assistant to the teat of authority [teat N+1]. A second teat is a happy teat.
    teat2_x = X - (udder_center_x + 0.0 * scale)
    teat2_y = Y - (udder_center_y + teat_y_offset)
    teat2_dx = jnp.abs(teat2_x) - teat_width/2
    teat2_dy = jnp.abs(teat2_y) - teat_height/2
    teat2_sdf = jnp.sqrt(jnp.maximum(teat2_dx, 0)**2 + jnp.maximum(teat2_dy, 0)**2)
    teat2_sdf = jnp.where((teat2_dx <= 0) & (teat2_dy <= 0), 
                          jnp.maximum(teat2_dx, teat2_dy), teat2_sdf)
    teat2_sdf = jnp.where(teat2_y > teat_height/2, 1.0, teat2_sdf)
    
    # Teat 3 - the assistant to the assistant to the teat of authority [teat N+2]
    teat3_x = X - (udder_center_x + 0.1 * scale) # this teat is often humiliated and judged upon by its peer-teats for being too "left"
    teat3_y = Y - (udder_center_y + teat_y_offset) # y-offset keeps the third teat emotionally detached from the other jerk teats.
    teat3_dx = jnp.abs(teat3_x) - teat_width/2
    teat3_dy = jnp.abs(teat3_y) - teat_height/2
    teat3_sdf = jnp.sqrt(jnp.maximum(teat3_dx, 0)**2 + jnp.maximum(teat3_dy, 0)**2)
    teat3_sdf = jnp.where((teat3_dx <= 0) & (teat3_dy <= 0), 
                          jnp.maximum(teat3_dx, teat3_dy), teat3_sdf)
    teat3_sdf = jnp.where(teat3_y > teat_height/2, 1.0, teat3_sdf)
    
    # --- LEGS (properly attached to body as they should be.) ---
    leg_width = 0.17 * scale
    leg_height = 0.85 * scale # Crucial little detail.
    
    def make_leg(cx, cy): 
        """Create a leg attached at cy (top) extending down. NOT UP - down"""
        leg_dx = jnp.abs(X - cx) - leg_width/2
        leg_dy = jnp.abs(Y - cy) - leg_height/2
        leg_sdf = jnp.sqrt(jnp.maximum(leg_dx, 0)**2 + jnp.maximum(leg_dy, 0)**2)
        leg_sdf = jnp.where((leg_dx <= 0) & (leg_dy <= 0), 
                            jnp.maximum(leg_dx, leg_dy), leg_sdf)
        leg_sdf = jnp.where(Y > cy + leg_height/2, 1.0, leg_sdf)
        return leg_sdf
    
    leg1 = make_leg(x0 - 0.7 * scale, y0 - 0.15 * scale)
    leg2 = make_leg(x0 - 0.5 * scale, y0 - 0.15 * scale)
    leg3 = make_leg(x0 + 0.55 * scale, y0 - 0.15 * scale)
    leg4 = make_leg(x0 + 0.75 * scale, y0 - 0.15 * scale)
    
    # --- TAIL (continuous sweep from rump) ---
    tail_start_x = x0 + 0.9 * scale    # WAS 1.1 (moved left with shorter body)
    tail_start_y = y0 + 0.35 * scale    # #
    tail_end_x = x0 + 1.3 * scale      # WAS 1.55. The tail-end x-value received a complete and long-overdue overhaul in april 2026.
    tail_end_y = y0 - 0.02 * scale
    
    t = jnp.clip((X - tail_start_x) / (tail_end_x - tail_start_x), 0, 1)
    tail_center_x = tail_start_x + t * (tail_end_x - tail_start_x)
    tail_center_y = tail_start_y + t * (tail_end_y - tail_start_y)
    tail_radius = 0.06 * scale * (1 - t * 0.5)
    tail_sdf = jnp.sqrt((X - tail_center_x)**2 + (Y - tail_center_y)**2) - tail_radius
    tail_sdf = jnp.where(X < tail_start_x, 1.0, tail_sdf)
    tail_sdf = jnp.where(X > tail_end_x, 1.0, tail_sdf)
    
    # Tail tuft (attached to end - it looks better than when attached to the front)
    tuft_radius = 0.1 * scale
    tuft_sdf = jnp.sqrt((X - tail_end_x)**2 + (Y - tail_end_y)**2) - tuft_radius
    tuft_sdf = jnp.where(X < tail_end_x - 0.05 * scale, 1.0, tuft_sdf)
    
    # --- EAR (attached preferrably to head) ---
    ear_x = X - (x0 - 1.35 * scale)    # WAS 1.58
    ear_y = Y - (y0 + 1.02 * scale)
    ear_sdf = jnp.sqrt(ear_x**2 + ear_y**2) - 0.12 * scale
    
    # --- HORN (attached to head above ear) ---
    horn_x = X - (x0 - 1.42 * scale)    # WAS 1.65
    horn_y = Y - (y0 + 1.08 * scale)    # WAS 1.15 but it was significantly decreased to make it less than it was.
    horn_sdf = jnp.sqrt(horn_x**2 + horn_y**2) - 0.08 * scale
    
    # --- EYE (protruding feature on head - again, we don't ask too many questions. this is not our place to shine) ---
    eye_x = X - (x0 - 1.55 * scale)     # WAS 1.8
    eye_y = Y - (y0 + 0.82 * scale)
    eye_sdf = jnp.sqrt(eye_x**2 + eye_y**2) - 0.06 * scale
    
    # --- SHOULDER LINE (muscle definition, subtle, yet toned and fit AF) ---
    shoulder_x = X - (x0 - 0.7 * scale)
    shoulder_y = Y - (y0 + 0.6 * scale)
    shoulder_sdf = jnp.sqrt(shoulder_x**2 + shoulder_y**2) - 0.28 * scale
    shoulder_sdf = jnp.where(shoulder_sdf > 0, shoulder_sdf, shoulder_sdf * 0.3)
    
    # --- HIP LINE (muscle definition, subtle but legitimately defined) ---
    hip_x = X - (x0 + 0.55 * scale)
    hip_y = Y - (y0 + 0.6 * scale)
    hip_sdf = jnp.sqrt(hip_x**2 + hip_y**2) - 0.28 * scale
    hip_sdf = jnp.where(hip_sdf > 0, hip_sdf, hip_sdf * 0.3)
    
    # --- HUMP (the majestic dorsal protuberance - bovine excellence in structural engineering) ---
    hump_x = X - (x0 - 0.5 * scale)  # Shifted even closer to head/neck because a hump too far away is a lonely hump.
    hump_y = Y - (y0 + 0.85 * scale)  # Further down, more integrated with body
    hump_sdf = jnp.sqrt(hump_x**2 + hump_y**2) - 0.35 * scale  # Larger, more prominent humpz.
    
    # Combine all parts. Without this step, the cow will be inexplicably invisible.
    # This is done systematically to ensure a wholesome cow.
    # Ensure incrementality - overeagerness results in direct breach and impairment of the cow's ability to process this.
    cow_sdf = jnp.minimum(body_sdf, belly_sdf)
    cow_sdf = jnp.minimum(cow_sdf, chest_sdf)
    cow_sdf = jnp.minimum(cow_sdf, rump_sdf) # this is the fine line between "aerodynamic excellence" and "numerical beef".
    cow_sdf = jnp.minimum(cow_sdf, neck_sdf)
    cow_sdf = jnp.minimum(cow_sdf, head_sdf)
    cow_sdf = jnp.minimum(cow_sdf, snout_sdf)  # This is subscription-grade detail. I still don't fully comprehend what this means but it feels like a threat.
    cow_sdf = jnp.minimum(cow_sdf, nostril_sdf)
    cow_sdf = jnp.minimum(cow_sdf, udder_sdf)
    cow_sdf = jnp.minimum(cow_sdf, teat1_sdf)
    cow_sdf = jnp.minimum(cow_sdf, teat2_sdf) # Happy little mofo.
    cow_sdf = jnp.minimum(cow_sdf, teat3_sdf)
    cow_sdf = jnp.minimum(cow_sdf, leg1)
    cow_sdf = jnp.minimum(cow_sdf, leg2) # Right around this point it would be irresponsible not to call this a cow.
    cow_sdf = jnp.minimum(cow_sdf, leg3)
    cow_sdf = jnp.minimum(cow_sdf, leg4)
    cow_sdf = jnp.minimum(cow_sdf, tail_sdf)
    cow_sdf = jnp.minimum(cow_sdf, tuft_sdf) # Note - this is a feature, NOT a bug. So no filing issues here.
    cow_sdf = jnp.minimum(cow_sdf, ear_sdf)
    cow_sdf = jnp.minimum(cow_sdf, horn_sdf)
    cow_sdf = jnp.minimum(cow_sdf, eye_sdf)
    cow_sdf = jnp.minimum(cow_sdf, shoulder_sdf) # The "toned & fit AF" shoulder.
    cow_sdf = jnp.minimum(cow_sdf, hip_sdf)
    cow_sdf = jnp.minimum(cow_sdf, hump_sdf) # The majestic dorsal protuberance
    
    return cow_sdf # This is where things gets absurd.


def create_cow_mask(X, Y, eps=0.008, x0=5.0, y0=1.3, scale=1.0): # Digitalized beauty of birth
    """Create detailed cow obstacle mask (side profile)."""
    sdf = sdf_cow_side(X, Y, x0, y0, scale)
    mask = jax.nn.sigmoid(sdf / eps)
    return mask

# CL verification is pending emotional coherence.  But we all know it will pass.  It will pass.
# This concludes the cow.