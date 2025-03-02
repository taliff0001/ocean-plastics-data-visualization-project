# Ocean Plastics Data Dictionary

## DOER Microplastics Database

| Variable | Description | Data Type | Example Values | Notes |
|----------|-------------|-----------|----------------|-------|
| mpID | Unique identifier for each entry | Integer | 97, 98, 96 | Primary key for the dataset |
| Continent | Continental region | String | "Africa", "Asia", "Europe" | Major geographic classification |
| Subdivision | Regional subdivision | String | "Bizerte", "South Florida" | 86 missing values (8.1%) |
| Country | Country name | String | "Tunisia", "China", "Japan" | 26 unique countries |
| System | Water system type | String | "Estuarine", "Marine", "Riverine" | 5 unique systems |
| Waterbody | Specific water body name | String | "Lagoon of Bizerte", "Yellow Sea" | Specific location descriptor |
| Zone Area | General zone classification | String | "Coastal", "Open water" | Classification of environmental zone |
| Tidal Zone | Tidal classification | String | "Subtidal", "Intertidal" | Position relative to tidal influence |
| Test Area | Specific sampling location | String | "Menzel Jemil", "Carrier Bay" | Site-specific identifier |
| Sample Time | Year of sampling | Integer | 2016, 2018 | Temporal reference |
| Sediment Sample Method | Collection methodology | String | "stainless steel spatula", "grab sampler" | Describes collection technique |
| Sediment Depth (cm) | Depth of sediment collection | String | "0-3 cm", "0-5 cm" | Vertical sampling information |
| Extract Sediment Wt | Weight of extracted sediment | String | "50 g dry", "500 g wet" | Sample quantity information |
| Extract Method | Extraction methodology | String | "1-density separated" | Process to isolate microplastics |
| Extract Chemical | Chemicals used in extraction | String | "sodium chloride (1.14 g/ml)" | Separation medium |
| Extract Oxidizer | Oxidizing agent used | String | "(none)", "H2O2" | For organic matter removal |
| No. of Extracts | Number of extraction repetitions | String | "1x", "3x" | Process repetitions |
| Inspection | Inspection method | String | "Visual inspection only" | Identification approach |
| mp range (µm) | Size range of microplastics | String | "10-5000", "63-5000" | Particle size bounds |
| mp/kg dw | Microplastic concentration | Float | 3682, 6628, 17921 | Particles per kg dry weight |
| MP Unit | Unit of measurement | String | "microplastic/kg sediment dry wt." | Standardized reporting unit |
| Dominant Shapes | Primary shapes observed | String | "fbr, frgmnt", "sphr" | Shape classifications |
| Dominant Size | Predominant size fraction | String | "<2 mm", "<1 mm" | Most common size category |
| Colors | Colors observed | String | "transparent, white, blue, red..." | Color classifications |
| Polymers | Polymer types identified | String | "(Not Applicable)", "PE, PP" | Chemical composition |
| N | Sample size | Float | 3, 5 | Number of replicates, 45 missing values |
| MP Stat | Statistical measure | String | "Mean", "Median" | Type of value reported |
| Notes | Additional information | String | Various notes | 582 missing values (54.7%) |
| Data Obtained From | Data source type | String | "interp_graph; Figure 2" | How data was extracted |
| Abbrev. Reference | Citation | String | "Abidli et al. 2017..." | Literature reference |
| DOI | Digital Object Identifier | String | "https://doi.org/10.1007/s11270-017-3439-9" | Publication link |

## Taiwan Beach Plastics Databases

### Particle Count Data (Longmen & Xialiao)

| Variable | Description | Data Type | Example Values | Notes |
|----------|-------------|-----------|----------------|-------|
| Date_YYYY-MM-DD | Sampling date | String | "2018-04-27", "2019-11-27" | Temporal reference |
| Country_Region | Country/region | String | "Taiwan" | Geographic location |
| Location_name | Beach name | String | "Longmen_Beach", "Xialiao_Beach" | Specific beach location |
| Location_lat | Latitude | Float | 25.02951, 25.21469 | Geographic coordinates |
| Location_lon | Longitude | Float | 121.9355, 121.65406 | Geographic coordinates |
| Transect | Sampling transect | String | "A", "B", "C" | Sampling design reference |
| Position | Position along transect | Integer | 1, 2, 3 | Spatial reference |
| Size_min_mm | Minimum size | Integer | 1, 5 | Lower size boundary (mm) |
| Size_max_mm | Maximum size | Integer | 5, 25 | Upper size boundary (mm) |
| Size_class | Size classification | String | "microplastics", "mesoplastics" | Size category |
| Particle_count | Number of particles | Float/Integer | 0, 3, 52 | Count of plastic particles |
| Particle_weight_g | Weight of particles | Float | 0, 0.001, 0.023 | Weight in grams |
| Weight_dry_sand_g | Weight of dry sand | Float | 1144.8, 1275 | Sample weight in grams |
| Beach_Zone | Beach zonation | String | "dune", "storm_line", "intertidal" | Ecological/geomorphological zone |

### Shape Count Data (Longmen & Xialiao)

| Variable | Description | Data Type | Example Values | Notes |
|----------|-------------|-----------|----------------|-------|
| [Identification columns] | Same as Particle Count Data | - | - | Same metadata as above |
| fragment | Count of fragments | Float/Integer | 0, 12, 45 | Irregular plastic pieces |
| foamed_plastic | Count of foam pieces | Float/Integer | 0, 23, 78 | Expanded polystyrene etc. |
| pellet | Count of pellets | Float/Integer | 0, 5, 12 | Pre-production plastic nurdles |
| foil | Count of foil/film | Float/Integer | 0, 1, 3 | Thin plastic sheets |
| fiber/fibers | Count of fibers | Float/Integer | 0, 2, 5 | Filaments (e.g., from textiles) |
| fishing_line | Count of fishing line | Float/Integer | 0, 1, 2 | Fishing-related filaments |
| cigarette_butt | Count of cigarette butts | Float/Integer | 0, 1 | Cigarette filters |
| rope | Count of rope pieces | Float/Integer | 0, 1 | Twisted fiber bundles |
| rubber | Count of rubber pieces | Float/Integer | 0, 1 | Rubber items |
| fabric | Count of fabric pieces | Float/Integer | 0, 1 | Textile fragments |
| unclear | Count of unidentified items | Float/Integer | 0, 1 | Unclassifiable items |

### Color Count Data (Longmen & Xialiao)

| Variable | Description | Data Type | Example Values | Notes |
|----------|-------------|-----------|----------------|-------|
| [Identification columns] | Same as Particle Count Data | - | - | Same metadata as above |
| no_color | Count of colorless items | Float/Integer | 0, 35, 112 | Transparent or clear items |
| black | Count of black items | Float/Integer | 0, 3, 15 | Black colored items |
| grey | Count of grey items | Float/Integer | 0, 2, 8 | Grey colored items |
| red_pink | Count of red/pink items | Float/Integer | 0, 1, 7 | Red or pink colored items |
| orange_brown_yellow | Count of orange/brown/yellow | Float/Integer | 0, 4, 12 | Warm color spectrum items |
| green | Count of green items | Float/Integer | 0, 2, 9 | Green colored items |
| blue | Count of blue items | Float/Integer | 0, 2, 8 | Blue colored items |
| purple | Count of purple items | Float/Integer | 0, 1, 2 | Purple colored items |

## Integrated Dataset (To Be Created)

| Variable | Description | Data Type | Notes |
|----------|-------------|-----------|-------|
| sample_id | Unique sample identifier | String | Combined ID for integrated dataset |
| source_dataset | Original data source | String | "DOER" or "Taiwan_Beaches" |
| location_name | Location name | String | Standardized location name |
| country | Country name | String | Country of sample |
| latitude | Latitude | Float | Geographic coordinate |
| longitude | Longitude | Float | Geographic coordinate |
| collection_date | Collection date | Date | Standardized date format |
| environment_type | Environment classification | String | "Marine", "Coastal", "Beach", etc. |
| zone | Specific zone | String | Beach zone or marine zone |
| size_class | Size classification | String | Standardized size categories |
| concentration_mp_kg | Microplastic concentration | Float | Standardized to particles per kg |
| shape_dominant | Dominant shape type | String | Primary shape category |
| color_dominant | Dominant color | String | Primary color category |
| sample_method | Sampling methodology | String | Collection approach |
| sample_depth_cm | Sampling depth | Float | Depth of collection |
| extraction_method | Extraction method | String | How plastics were separated |
| notes | Additional information | String | Context and special observations |
| reference | Data reference | String | Citation or study reference |
