#!/usr/bin/env python3
"""Automated annotation utilities for error case analysis.

Extracts linguistic and contextual features from event notes and ACLED metadata
to assist qualitative analysis of false legitimization/illegitimization errors.

Based on conflict analysis literature and ACLED methodology:
- Provenance: source type inference from SOURCE_SCALE and SOURCE columns
- Verb intensity: lexical analysis of violence-related verbs  
- Casualty counts: detection from FATALITIES column and note text
- Passive voice: syntactic pattern detection
- Ambiguous actor: heuristics for unclear actor attribution
"""
from __future__ import annotations
import re
from typing import Optional

# ============================================================================
# PROVENANCE CLASSIFICATION
# ============================================================================
# Based on ACLED source scale taxonomy and media bias literature
# Enhanced with Africa-specific media outlets from Cameroon, Nigeria, and regional sources

# State-affiliated media patterns (government-controlled or pro-government)
STATE_MEDIA_PATTERNS = [
    # Government-affiliated sources
    r'\bgovernment\s+(spokesman|official|source)',
    r'\bministry\s+of',
    r'\bstate\s+media',
    r'\bofficial\s+(statement|press|release)',
    r'\bpresidential',
    r'\bmilitary\s+(spokesman|source|official)',
    r'\barmy\s+(spokesman|source)',
    r'\bpolice\s+(spokesman|source)',
    # Cameroon state/government-aligned
    r'\bCRTV\b',  # Cameroon Radio Television (state broadcaster)
    r'\bCameroon\s+Tribune\b',  # State-owned newspaper
    r'\bCameroon\s+News\s+Agency\b',  # CNA - state news agency
    # Nigeria state/government-aligned  
    r'\bNTA\b',  # Nigerian Television Authority (state)
    r'\bFRCN\b',  # Federal Radio Corporation of Nigeria (state)
    r'\bNews\s+Agency\s+of\s+Nigeria\b',  # NAN - state news agency
    r'\bNAN\b',
    # Other African state media
    r'\bAlgeria\s+Press\s+Service\b',  # APS - Algerian state agency
    r'\bAPS\b',
    r'\bXinhua\b',  # Chinese state media (often carries government statements)
    r'\bAnadolu\b',  # Turkish state media
]

# Independent/international media patterns
INDEPENDENT_MEDIA_PATTERNS = [
    # Major international wire services
    r'\bReuters\b',
    r'\bAFP\b',
    r'\bAP\b',
    r'\bAssociated\s+Press\b',
    # International broadcasters
    r'\bBBC\b',
    r'\bVOA\b',  # Voice of America
    r'\bRFI\b',  # Radio France Internationale
    r'\bAl\s+Jazeera\b',
    r'\bFrance\s+24\b',
    r'\bDW\b',  # Deutsche Welle
    # Major international newspapers
    r'\bThe\s+Guardian\b',
    r'\bNew\s+York\s+Times\b',
    r'\bWashington\s+Post\b',
    r'\bLe\s+Monde\b',
    # International research/monitoring organizations
    r'\bInternational\s+Crisis\s+Group\b',
    r'\bICG\b',
    r'\bUNOCHA\b',
    r'\bUNHCR\b',
    r'\bAmnesty\s+International\b',
    r'\bHuman\s+Rights\s+Watch\b',
    r'\bHRW\b',
    r'\bCentre\s+for\s+Human\s+Rights\b',
]

# Nigerian independent/private media
NIGERIA_INDEPENDENT_PATTERNS = [
    r'\bDaily\s+Trust\b',
    r'\bDaily\s+Post\b',
    r'\bNigeria\s+Punch\b',
    r'\bPunch\b',
    r'\bSahara\s+Reporters\b',
    r'\bDaily\s+Independent\b',
    r'\bSun\s+\(Nigeria\)\b',
    r'\bVanguard\b',
    r'\bBlueprint\b',
    r'\bPremium\s+Times\b',
    r'\bDaily\s+Leadership\b',
    r'\bGuardian\s+\(Nigeria\)\b',
    r'\bHumAngle\b',
    r'\bNew\s+Telegraph\b',
    r'\bThe\s+Cable\b',
    r'\bDaily\s+Champion\b',
    r'\bCKN\s+Nigeria\b',
    r'\bInside\s+Arewa\b',
    r'\bEONS\s+Intelligence\b',
    r'\bCWC\s+\(Nigeria\)\b',
]

# Cameroonian independent/private media
CAMEROON_INDEPENDENT_PATTERNS = [
    r'\bMimi\s+Mefo\b',  # Independent journalist/outlet
    r'\bSembe\s+TV\b',
    r'\bHumanity\s+Purpose\b',
    r"\bL'Oeil\b",
    r'\bCamer\.be\b',
    r'\bCameroon\s+Online\b',
    r'\bJournal\s+du\s+Cameroun\b',
    r'\bACTU\s+Cameroun\b',
]

# North African media (Algeria, Morocco, etc.)
NORTH_AFRICA_MEDIA_PATTERNS = [
    r'\bEchorouk\b',
    r'\bAkher\s+Saa\b',
    r'\bEl\s+Khabar\b',
    r'\bEl\s+Watan\b',
    r'\bLe\s+Soir\s+d\'Algerie\b',
    r'\bLe\s+Quotidien\s+d\'Oran\b',
    r'\bDjazairess\b',
    r"\bL'Expression\b",
    r'\bEl\s+Massa\b',
    r'\bTSA\s+Algerie\b',
    r'\bEl\s+Djoumhouria\b',
    r'\bYabiladi\b',
    r'\bDjelfa\s+Info\b',
    r'\bLe360\b',
    r'\bMaghress\b',
    r'\bHespress\b',
    r'\bEnnahar\s+Online\b',
]

# Local/community and social media patterns
LOCAL_MEDIA_PATTERNS = [
    r'\blocal\s+(media|source|journalist)',
    r'\bcommunity\s+source',
    r'\bwitness',
    r'\bresident',
    r'\bvillager',
    r'\bUndisclosed\s+Source\b',
    r'\bLocal\s+partner\b',
]

# Social media / new media patterns
SOCIAL_MEDIA_PATTERNS = [
    r'\bTwitter\b',
    r'\bFacebook\b',
    r'\bTelegram\b',
    r'\bWhatsApp\b',
    r'\bTikTok\b',
    r'\bInstagram\b',
    r'\bNew\s+media\b',
]

# Risk/security analysis firms
SECURITY_ANALYSIS_PATTERNS = [
    r'\bRisk\s+and\s+Strategic\s+Management\b',
    r'\bRSM\b',
    r'\bControl\s+Risks\b',
    r'\bJanes\b',
    r'\bSITREP\b',
]


def classify_provenance(source: Optional[str], source_scale: Optional[str], notes: Optional[str]) -> str:
    """Classify source provenance based on ACLED metadata and note content.
    
    Returns one of: State-affiliated, Independent-International, Independent-National, 
                    Security-Analysis, Social-Media, Local/Community, Unknown
    """
    combined_text = ' '.join(filter(None, [str(source or ''), str(notes or '')]))
    
    # Check source patterns in order of specificity
    state_match = any(re.search(p, combined_text, re.I) for p in STATE_MEDIA_PATTERNS)
    intl_match = any(re.search(p, combined_text, re.I) for p in INDEPENDENT_MEDIA_PATTERNS)
    nga_match = any(re.search(p, combined_text, re.I) for p in NIGERIA_INDEPENDENT_PATTERNS)
    cmr_match = any(re.search(p, combined_text, re.I) for p in CAMEROON_INDEPENDENT_PATTERNS)
    na_match = any(re.search(p, combined_text, re.I) for p in NORTH_AFRICA_MEDIA_PATTERNS)
    local_match = any(re.search(p, combined_text, re.I) for p in LOCAL_MEDIA_PATTERNS)
    social_match = any(re.search(p, combined_text, re.I) for p in SOCIAL_MEDIA_PATTERNS)
    security_match = any(re.search(p, combined_text, re.I) for p in SECURITY_ANALYSIS_PATTERNS)
    
    # Determine primary classification
    # State media takes precedence if detected
    if state_match and not (intl_match or nga_match or cmr_match):
        return 'State-affiliated'
    
    # Security/risk analysis firms
    if security_match:
        return 'Security-Analysis'
    
    # International independent media
    if intl_match:
        return 'Independent-International'
    
    # National independent media (Nigeria or Cameroon)
    if nga_match or cmr_match or na_match:
        return 'Independent-National'
    
    # Social media sources
    if social_match and not (nga_match or cmr_match or intl_match):
        return 'Social-Media'
    
    # Local/community sources
    if local_match:
        return 'Local/Community'
    
    # Use SOURCE_SCALE as fallback
    if source_scale:
        scale = str(source_scale).lower()
        if 'international' in scale:
            return 'Independent-International'
        elif 'national' in scale:
            return 'Independent-National'
        elif 'regional' in scale:
            return 'Regional'
        elif 'subnational' in scale or 'local' in scale:
            return 'Local/Community'
        elif 'new media' in scale:
            return 'Social-Media'
    
    return 'Unknown'


# ============================================================================
# VERB INTENSITY CLASSIFICATION
# ============================================================================
# Violence verb taxonomy from conflict linguistics literature

HIGH_INTENSITY_VERBS = [
    # Lethal violence
    r'\bkill(ed|ing|s)?\b', r'\bmurder(ed|ing|s)?\b', r'\bslaughter(ed|ing|s)?\b',
    r'\bmassacre[ds]?\b', r'\bexecut(ed|ing|ion|es)?\b', r'\bassassinat(ed|ing|ion|es)?\b',
    r'\bbehead(ed|ing|s)?\b', r'\bbutcher(ed|ing|s)?\b', r'\bslain\b',
    # Severe physical violence  
    r'\btortur(ed|ing|es?)?\b', r'\bmaim(ed|ing|s)?\b', r'\bmutilat(ed|ing|ion|es)?\b',
    r'\brape[ds]?\b', r'\bgang-?rape[ds]?\b',
    # Explosive/heavy weapons
    r'\bbomb(ed|ing|s)?\b', r'\bshell(ed|ing|s)?\b', r'\bblast(ed|ing|s)?\b',
    r'\bdetonat(ed|ing|ion|es)?\b', r'\bexplod(ed|ing|es)?\b',
    # Mass harm
    r'\braz(ed|ing|es)?\b', r'\bdestroy(ed|ing|s)?\b', r'\bannihilat(ed|ing|ion|es)?\b',
]

MEDIUM_INTENSITY_VERBS = [
    # Armed engagement
    r'\battack(ed|ing|s)?\b', r'\bshoot(ing|s)?\b', r'\bshot\b', r'\bfir(ed|ing|es)?\b',
    r'\bstrike[ds]?\b', r'\bstruck\b', r'\bambush(ed|ing|es)?\b',
    # Physical violence
    r'\bbeat(en|ing|s)?\b', r'\bassault(ed|ing|s)?\b', r'\bstab(bed|bing|s)?\b',
    r'\bwound(ed|ing|s)?\b', r'\binjur(ed|ing|y|ies)?\b',
    # Property destruction
    r'\bburn(ed|t|ing|s)?\b', r'\bloot(ed|ing|s)?\b', r'\bvandaliz(ed|ing|es)?\b',
    r'\braid(ed|ing|s)?\b',
    # Forcible actions
    r'\babduct(ed|ing|ion|s)?\b', r'\bkidnap(ped|ping|s)?\b', r'\bseiz(ed|ing|es)?\b',
]

LOW_INTENSITY_VERBS = [
    # Demonstrations/protests
    r'\bprotest(ed|ing|s)?\b', r'\bdemonstr(ated|ating|ation|ations)?\b',
    r'\brall(y|ied|ying|ies)\b', r'\bmarch(ed|ing|es)?\b',
    # Non-violent conflict
    r'\bclash(ed|ing|es)?\b', r'\bconfront(ed|ing|ation|s)?\b',
    r'\bdispers(ed|ing|al|es)?\b', r'\bblock(ed|ing|ade|s)?\b',
    # Threats/intimidation
    r'\bthreat(en|ened|ening|s)?\b', r'\bintimidat(ed|ing|ion|es)?\b',
    r'\bharass(ed|ing|ment|es)?\b', r'\bdetain(ed|ing|s)?\b', r'\barrest(ed|ing|s)?\b',
]


def classify_verb_intensity(notes: Optional[str]) -> str:
    """Classify verb intensity in event description.
    
    Returns: High, Medium, Low, or Unknown
    """
    if not notes:
        return 'Unknown'
    
    text = str(notes).lower()
    
    high_count = sum(1 for p in HIGH_INTENSITY_VERBS if re.search(p, text))
    medium_count = sum(1 for p in MEDIUM_INTENSITY_VERBS if re.search(p, text))
    low_count = sum(1 for p in LOW_INTENSITY_VERBS if re.search(p, text))
    
    # Prioritize by intensity level and count
    if high_count >= 1:
        return 'High'
    elif medium_count >= 1:
        return 'Medium'
    elif low_count >= 1:
        return 'Low'
    
    return 'Unknown'


# ============================================================================
# CASUALTY DETECTION
# ============================================================================

CASUALTY_PATTERNS = [
    r'\b(\d+)\s*(people|persons?|civilians?|soldiers?|militants?|fighters?|victims?)\s*(were\s+)?(killed|died|dead|slain)',
    r'\b(killed|dead|died|slain)\s*(\d+)',
    r'\bfatalit(y|ies)\b',
    r'\bdeath\s*toll\b',
    r'\bcasualt(y|ies)\b',
    r'\bbod(y|ies)\s*(found|recovered|discovered)',
    r'\b(dozens?|hundreds?|scores?)\s*(of\s+)?(people|civilians?|victims?)?\s*(killed|dead|died)',
    r'\bmass\s*(grave|killing|casualt)',
]


def detect_casualties(fatalities: Optional[float], notes: Optional[str]) -> str:
    """Detect whether casualties are mentioned.
    
    Returns: Yes (N), Yes (unspecified), or No
    """
    # Check ACLED FATALITIES column first
    if fatalities is not None:
        try:
            fat_count = int(float(fatalities))
            if fat_count > 0:
                return f'Yes ({fat_count})'
        except (ValueError, TypeError):
            pass
    
    # Check note text for casualty mentions
    if notes:
        text = str(notes).lower()
        for pattern in CASUALTY_PATTERNS:
            match = re.search(pattern, text)
            if match:
                # Try to extract number
                num_match = re.search(r'\d+', match.group())
                if num_match:
                    return f'Yes ({num_match.group()})'
                return 'Yes (unspecified)'
    
    return 'No'


# ============================================================================
# PASSIVE VOICE DETECTION
# ============================================================================
# Enhanced patterns for detecting passive voice and agent obscuring in conflict reporting

# Passive voice constructions (be + past participle)
PASSIVE_BE_PATTERNS = [
    # Standard passive with "was/were"
    r'\bwas\s+\w+ed\b',
    r'\bwere\s+\w+ed\b',
    r'\bhas\s+been\s+\w+ed\b',
    r'\bhave\s+been\s+\w+ed\b',
    r'\bhad\s+been\s+\w+ed\b',
    r'\bbeing\s+\w+ed\b',
    r'\bbeen\s+\w+ed\b',
    # Irregular past participles
    r'\bwas\s+\w+en\b',  # was taken, was beaten, was given
    r'\bwere\s+\w+en\b',
    r'\bhas\s+been\s+\w+en\b',
    r'\bhave\s+been\s+\w+en\b',
    r'\bwas\s+(shot|hit|cut|put|shut|hurt|set)\b',
    r'\bwere\s+(shot|hit|cut|put|shut|hurt|set)\b',
    # Specific conflict-related passive constructions
    r'\bwas\s+(killed|murdered|slain|executed|assassinated)\b',
    r'\bwere\s+(killed|murdered|slain|executed|assassinated)\b',
    r'\bwas\s+(attacked|ambushed|raided|assaulted|abducted)\b',
    r'\bwere\s+(attacked|ambushed|raided|assaulted|abducted)\b',
    r'\bwas\s+(burned|burnt|destroyed|looted|razed)\b',
    r'\bwere\s+(burned|burnt|destroyed|looted|razed)\b',
    r'\bwas\s+(injured|wounded|hurt|maimed)\b',
    r'\bwere\s+(injured|wounded|hurt|maimed)\b',
    r'\bwas\s+(kidnapped|captured|seized|detained|arrested)\b',
    r'\bwere\s+(kidnapped|captured|seized|detained|arrested)\b',
    r'\bwas\s+(bombed|shelled|fired\s+upon|shot\s+at)\b',
    r'\bwere\s+(bombed|shelled|fired\s+upon|shot\s+at)\b',
]

# Agent obscuring phrases (often accompany passive voice)
AGENT_OBSCURING_PATTERNS = [
    r'\bby\s+unknown\s+(assailants?|gunmen|attackers?|perpetrators?|actors?|persons?|individuals?)',
    r'\bby\s+unidentified\s+(assailants?|gunmen|attackers?|perpetrators?|actors?|persons?|individuals?)',
    r'\bby\s+suspected\s+(militants?|gunmen|attackers?|terrorists?|criminals?)',
    r'\bby\s+armed\s+(men|persons?|individuals?|group)',
    r'\bby\s+unknown\s+persons?\b',
    r'\bby\s+gunmen\b',
    r'\bby\s+assailants?\b',
    r'\bby\s+attackers?\b',
]

# Epistemic hedging (uncertainty markers often used with passive)
HEDGING_PATTERNS = [
    r'\ballegedly\b',
    r'\breportedly\b',
    r'\bsuspected(ly)?\b',
    r'\bpurportedly\b',
    r'\bapparently\b',
    r'\bbelieved\s+to\s+(be|have)\b',
    r'\bthought\s+to\s+(be|have)\b',
    r'\bsaid\s+to\s+(be|have)\b',
    r'\bclaimed\s+to\b',
    r'\baccording\s+to\s+(reports?|sources?)\b',
]

# Nominalization patterns (converting verbs to nouns, obscures agent)
NOMINALIZATION_PATTERNS = [
    r'\bthe\s+(killing|murder|assassination|execution)\s+of\b',
    r'\bthe\s+(attack|raid|assault|ambush)\s+on\b',
    r'\bthe\s+(destruction|burning|looting)\s+of\b',
    r'\bthe\s+(abduction|kidnapping|capture)\s+of\b',
    r'\bthe\s+(bombing|shelling)\s+of\b',
    r'\bduring\s+(the\s+)?(attack|raid|assault|clash)',
    r'\bin\s+(the\s+)?(attack|raid|assault|incident)',
]

# Active voice with clear attribution (counter-patterns)
ACTIVE_PATTERNS = [
    # Clear subject + violent verb patterns
    r'\b(military|army|police|soldiers|troops|forces)\s+(killed|attacked|shot|fired|bombed|raided|stormed|destroyed|arrested|detained|dispersed|opened\s+fire)',
    r'\b(militants?|fighters?|gunmen|attackers?|rebels?|insurgents?)\s+(killed|attacked|shot|fired|bombed|raided|stormed)',
    r'\b(group|gang|mob)\s+(killed|attacked|shot|assaulted)',
    # Named actor patterns
    r'\b[A-Z][a-z]+\s+(Forces|Army|Military|Police)\s+(killed|attacked|shot|arrested|detained)',
    r'\bBoko\s+Haram\s+(killed|attacked|bombed|raided)',
    r'\bISWAP\s+(killed|attacked|bombed|raided)',
    r'\bISIS\s+(killed|attacked|bombed)',
    r'\bAl[\s-]?(Qaeda|Shabaab)\s+(killed|attacked|bombed)',
    # Explicit agent constructions
    r'\b(they|he|she)\s+(killed|attacked|shot|fired|bombed)',
    # Additional active voice patterns
    r'\b(police|soldiers|troops|military|army)\s+(opened\s+fire|launched|conducted|carried\s+out)',
    r'\b(gunmen|attackers|militants)\s+(opened\s+fire|launched|carried\s+out)',
    r'\b\w+\s+(arrested|detained|seized|captured)\s+\d*\s*(suspects?|persons?|people|militants?|members?)',
]


def detect_passive_voice(notes: Optional[str]) -> str:
    """Detect passive voice usage in event description.
    
    Returns: Yes, No, Mixed, or Unknown
    
    Analysis considers:
    - Passive voice constructions (be + past participle)
    - Agent obscuring phrases
    - Epistemic hedging (allegedly, reportedly, etc.)
    - Nominalization (the killing of, the attack on)
    - Active voice with clear attribution
    """
    if not notes:
        return 'Unknown'
    
    text = str(notes)
    
    # Count different pattern types
    passive_be_count = sum(1 for p in PASSIVE_BE_PATTERNS if re.search(p, text, re.I))
    agent_obscure_count = sum(1 for p in AGENT_OBSCURING_PATTERNS if re.search(p, text, re.I))
    hedging_count = sum(1 for p in HEDGING_PATTERNS if re.search(p, text, re.I))
    nominal_count = sum(1 for p in NOMINALIZATION_PATTERNS if re.search(p, text, re.I))
    active_count = sum(1 for p in ACTIVE_PATTERNS if re.search(p, text, re.I))
    
    # Calculate passive score (weighted)
    passive_score = passive_be_count + (agent_obscure_count * 1.5) + hedging_count + nominal_count
    active_score = active_count * 1.5  # Weight active voice slightly higher
    
    # Determine classification
    if passive_score > 0 and active_score > 0:
        # Both present - check relative balance
        if passive_score > active_score * 1.5:
            return 'Yes'
        elif active_score > passive_score * 1.5:
            return 'No'
        else:
            return 'Mixed'
    elif passive_score >= 1:
        return 'Yes'
    elif active_score >= 1:
        return 'No'
    
    # No clear patterns found
    return 'Unknown'


# ============================================================================
# AMBIGUOUS ACTOR DETECTION
# ============================================================================

AMBIGUOUS_ACTOR_PATTERNS = [
    r'\bunknown\s+(assailants?|gunmen|attackers?|perpetrators?|actors?|militants?)',
    r'\bunidentified\s+(assailants?|gunmen|attackers?|perpetrators?|actors?|militants?)',
    r'\bsuspected\s+(militants?|gunmen|attackers?|terrorists?)',
    r'\balleged(ly)?\s+(militants?|gunmen|attackers?)',
    r'\b(gunmen|attackers?|assailants?)\s+believed\s+to\s+be',
    r'\battribut(ed|ion)\s+to',
    r'\bclaimed\s+responsibility',
    r'\bno\s+(group|one)\s+(has\s+)?claimed',
    r'\bunclear\s+(who|which)',
    r'\b(disputed|conflicting)\s+(reports?|accounts?)',
]


def detect_ambiguous_actor(notes: Optional[str], actor_norm: Optional[str]) -> str:
    """Detect whether actor attribution is ambiguous.
    
    Returns: Yes, No, or Partial
    """
    if not notes:
        return 'Unknown'
    
    text = str(notes).lower()
    actor = str(actor_norm or '').lower()
    
    # Check for explicit ambiguity markers
    ambig_count = sum(1 for p in AMBIGUOUS_ACTOR_PATTERNS if re.search(p, text))
    
    # Check if actor name itself suggests ambiguity
    actor_ambiguous = any(term in actor for term in ['unknown', 'unidentified', 'suspected', 'generic'])
    
    if ambig_count >= 2 or actor_ambiguous:
        return 'Yes'
    elif ambig_count == 1:
        return 'Partial'
    
    return 'No'


# ============================================================================
# MAIN ANNOTATION FUNCTION
# ============================================================================

def auto_annotate_row(row: dict) -> dict:
    """Generate automated annotations for a single error case row.
    
    Args:
        row: Dictionary with at minimum 'notes' key, optionally:
             'source', 'source_scale', 'fatalities', 'actor_norm'
    
    Returns:
        Dictionary with annotation_* keys filled in
    """
    notes = row.get('notes') or row.get('NOTES')
    source = row.get('source') or row.get('SOURCE')
    source_scale = row.get('source_scale') or row.get('SOURCE_SCALE')
    fatalities = row.get('fatalities') or row.get('FATALITIES')
    actor_norm = row.get('actor_norm')
    
    return {
        'annotation_provenance': classify_provenance(source, source_scale, notes),
        'annotation_ambiguous_actor': detect_ambiguous_actor(notes, actor_norm),
        'annotation_verb_intensity': classify_verb_intensity(notes),
        'annotation_casualty_counts': detect_casualties(fatalities, notes),
        'annotation_passive_voice': detect_passive_voice(notes),
        'annotation_notes': '[auto-generated]',
    }


def auto_annotate_dataframe(df, original_data=None):
    """Apply automated annotations to a DataFrame of error cases.
    
    Args:
        df: DataFrame with error cases (must have 'event_id' and 'notes' columns)
        original_data: Optional DataFrame with ACLED source data (SOURCE, SOURCE_SCALE, FATALITIES)
                      If provided, merges on event_id to get additional metadata
    
    Returns:
        DataFrame with annotation_* columns populated
    """
    import pandas as pd
    
    df = df.copy()
    
    # If original ACLED data provided, merge to get SOURCE, SOURCE_SCALE, FATALITIES
    if original_data is not None:
        # Normalize column names
        orig_cols = original_data.columns.str.upper()
        original_data.columns = orig_cols
        
        # Find event ID column
        event_id_col = None
        for col in ['EVENT_ID_CNTY', 'EVENT_ID', 'event_id']:
            if col.upper() in orig_cols:
                event_id_col = col.upper()
                break
        
        if event_id_col:
            merge_cols = [event_id_col]
            for col in ['SOURCE', 'SOURCE_SCALE', 'FATALITIES']:
                if col in orig_cols:
                    merge_cols.append(col)
            
            if len(merge_cols) > 1:
                df = df.merge(
                    original_data[merge_cols].drop_duplicates(),
                    left_on='event_id',
                    right_on=event_id_col,
                    how='left'
                )
    
    # Apply auto-annotation to each row
    annotations = df.apply(lambda row: pd.Series(auto_annotate_row(row.to_dict())), axis=1)
    
    # Update annotation columns (don't overwrite if already filled)
    for col in annotations.columns:
        if col not in df.columns:
            df[col] = annotations[col]
        else:
            # Only fill where empty
            mask = (df[col].isna()) | (df[col] == '')
            df.loc[mask, col] = annotations.loc[mask, col]
    
    return df
