########################
# Extraction
########################
# These are utilities to support structured data extraction tasks through the OpenAI function calling interface (wrapped by `aiextract`)
#
# There are potential formats: 1) JSON-based for OpenAI compatible APIs, 2) XML-based for Anthropic compatible APIs (used also by Hermes-2-Pro model). 
#

######################
# 1) OpenAI / JSON format
######################

to_json_type(s::Type{<:AbstractString}) = "string"
to_json_type(n::Type{<:Real}) = "number"
to_json_type(n::Type{<:Integer}) = "integer"
to_json_type(b::Type{Bool}) = "boolean"
to_json_type(t::Type{<:Union{Missing, Nothing}}) = "null"
to_json_type(t::Type{<:Any}) = "string" # object?

has_null_type(T::Type{Missing}) = true
has_null_type(T::Type{Nothing}) = true
has_null_type(T::Type) = T isa Union && any(has_null_type, Base.uniontypes(T))
## For required fields, only Nothing is considered a null type (and be easily parsed by JSON3)
is_required_field(T::Type{Nothing}) = false
function is_required_field(T::Type)
    if T isa Union
        all(is_required_field, Base.uniontypes(T))
    else
        true
    end
end

# Remove null types from Union etc.
remove_null_types(T::Type{Missing}) = Any
remove_null_types(T::Type{Nothing}) = Any
remove_null_types(T::Type{Union{Nothing, Missing}}) = Any
function remove_null_types(T::Type)
    T isa Union ? Union{filter(!has_null_type, Base.uniontypes(T))...} : T
end

function extract_docstring(type::Type; max_description_length::Int = 100)
    ## plain struct has supertype Any
    ## we ignore the ones that are subtypes for now (to prevent picking up Dicts, etc.)
    if supertype(type) == Any
        docs = Docs.doc(type) |> string
        if !occursin("No documentation found.\n\n", docs)
            return first(docs, max_description_length)
        end
    end
    return ""
end

function to_json_schema(orig_type; max_description_length::Int = 100)
    schema = Dict{String, Any}()
    type = remove_null_types(orig_type)
    if isstructtype(type)
        schema["type"] = "object"
        schema["properties"] = Dict{String, Any}()
        ## extract the field names and types
        required_types = String[]
        for (field_name, field_type) in zip(fieldnames(type), fieldtypes(type))
            schema["properties"][string(field_name)] = to_json_schema(
                remove_null_types(field_type);
                max_description_length)
            ## Hack: no null type (Nothing, Missing) implies it it is a required field
            is_required_field(field_type) && push!(required_types, string(field_name))
        end
        !isempty(required_types) && (schema["required"] = required_types)
        ## docstrings
        docs = extract_docstring(type; max_description_length)
        !isempty(docs) && (schema["description"] = docs)
    else
        schema["type"] = to_json_type(type)
    end
    return schema
end
function to_json_schema(type::Type{<:AbstractString}; max_description_length::Int = 100)
    Dict("type" => to_json_type(type))
end
function to_json_schema(type::Type{T};
        max_description_length::Int = 100) where {T <:
                                                  Union{AbstractSet, Tuple, AbstractArray}}
    element_type = eltype(type)
    return Dict("type" => "array",
        "items" => to_json_schema(remove_null_types(element_type)))
end
function to_json_schema(type::Type{<:Enum}; max_description_length::Int = 100)
    enum_options = Base.Enums.namemap(type) |> values .|> string
    return Dict("type" => "string",
        "enum" => enum_options)
end
function to_json_schema(type::Type{<:AbstractDict}; max_description_length::Int = 100)
    throw(ArgumentError("Dicts are not supported yet as we cannot analyze their keys/values on a type-level. Use a nested Struct instead!"))
end

"""
    function_call_signature(datastructtype::Struct; max_description_length::Int = 100)

Extract the argument names, types and docstrings from a struct to create the function call signature in JSON schema.

You must provide a Struct type (not an instance of it) with some fields.

Note: Fairly experimental, but works for combination of structs, arrays, strings and singletons.

# Tips
- You can improve the quality of the extraction by writing a helpful docstring for your struct (or any nested struct). It will be provided as a description. 
 You can even include comments/descriptions about the individual fields.
- All fields are assumed to be required, unless you allow null values (eg, `::Union{Nothing, Int}`). Fields with `Nothing` will be treated as optional.
- Missing values are ignored (eg, `::Union{Missing, Int}` will be treated as Int). It's for broader compatibility and we cannot deserialize it as easily as `Nothing`.

# Example

Do you want to extract some specific measurements from a text like age, weight and height?
You need to define the information you need as a struct (`return_type`):
```
struct MyMeasurement
    age::Int
    height::Union{Int,Nothing}
    weight::Union{Nothing,Float64}
end
signature = function_call_signature(MyMeasurement)
#
# Dict{String, Any} with 3 entries:
#   "name"        => "MyMeasurement_extractor"
#   "parameters"  => Dict{String, Any}("properties"=>Dict{String, Any}("height"=>Dict{String, Any}("type"=>"integer"), "weight"=>Dic…
#   "description" => "Represents person's age, height, and weight\n"
```

You can see that only the field `age` does not allow null values, hence, it's "required".
While `height` and `weight` are optional.
```
signature["parameters"]["required"]
# ["age"]
```

If there are multiple items you want to extract, define a wrapper struct to get a Vector of `MyMeasurement`:
```
struct MyMeasurementWrapper
    measurements::Vector{MyMeasurement}
end

Or if you want your extraction to fail gracefully when data isn't found, use `MaybeExtract{T}` wrapper (inspired by Instructor package!):
```
using PromptingTools: MaybeExtract

type = MaybeExtract{MyMeasurement}
# Effectively the same as:
# struct MaybeExtract{T}
#     result::Union{T, Nothing}
#     error::Bool // true if a result is found, false otherwise
#     message::Union{Nothing, String} // Only present if no result is found, should be short and concise
# end

# If LLM extraction fails, it will return a Dict with `error` and `message` fields instead of the result!
msg = aiextract("Extract measurements from the text: I am giraffe", type)

#
# Dict{Symbol, Any} with 2 entries:
# :message => "Sorry, this feature is only available for humans."
# :error   => true
```
That way, you can handle the error gracefully and get a reason why extraction failed.
"""
function function_call_signature(datastructtype::Type; max_description_length::Int = 100)
    !isstructtype(datastructtype) &&
        error("Only Structs are supported (provided type: $datastructtype")
    ## Standardize the name
    name = string(datastructtype, "_extractor") |>
           x -> replace(x, r"[^0-9A-Za-z_-]" => "") |> x -> first(x, 64)
    schema = Dict{String, Any}("name" => name,
        "parameters" => to_json_schema(datastructtype; max_description_length))
    ## docstrings
    docs = extract_docstring(datastructtype; max_description_length)
    !isempty(docs) && (schema["description"] = docs)
    ## remove duplicated Struct docstring in schema
    if haskey(schema["parameters"], "description") &&
       schema["parameters"]["description"] == docs
        delete!(schema["parameters"], "description")
    end
    return schema
end

######################
# 2) Anthropic / XML format
######################

"""
Simple template to add to the System Message when doing data extraction with Anthropic models.

It has 2 placeholders: `tool_name`, `tool_description` and `tool_parameters` that are filled with the tool's name, description and parameters.
Source: https://docs.anthropic.com/claude/docs/functions-external-tools
"""
ANTHROPIC_TOOL_PROMPT = """
  In this environment you have access to a specific tool you MUST use to answer the user's question.

  You should call it like this:
  <function_calls>
  <invoke>
  <tool_name>\$TOOL_NAME</tool_name>
  <parameters>
  <\$PARAMETER_NAME>\$PARAMETER_VALUE</\$PARAMETER_NAME>
  ...
  </parameters>
  </invoke>
  </function_calls>

  Here are the tools available:
  <tools>
  {{tool_definition}}
  </tools>
  """
ANTHROPIC_TOOL_PROMPT_LIST_EXTRA = """
  For any List[] types, include multiple <\$PARAMETER_NAME>\$PARAMETER_VALUE</\$PARAMETER_NAME> tags for each item in the list. XML tags should only contain the name of the parameter.
  """

"Converts from type:array"
function to_xml_schema(::Type{Vector}, properties::Dict)
    outputs = String[]
    subtype = properties["type"]
    if subtype == "object"
        ## add description
        push!(outputs, "<type>List[Dict]</type>")
        append!(outputs, to_xml_schema(Dict, properties["properties"]))
    elseif subtype == "array"
        @warn "Nested arrays are not supported in current XML parser. Expect errors."
        push!(outputs, "<type>List[...]</type>")
        append!(outputs, to_xml_schema(Vector, properties["items"]))
    else
        ## eg, integer etc
        push!(outputs, "<type>List[$subtype]</type>")
    end
    return outputs
end
"Converts from type:string/enum"
function to_xml_schema(::Type{Enum}, properties::Dict)
    outputs = String[]
    subtype = properties["type"]
    ## eg, integer etc
    push!(outputs, "<type>$subtype</type>")
    push!(outputs, "<values>")
    append!(outputs, ["<value>$val</value>" for val in properties["enum"]])
    push!(outputs, "</values>")
    return outputs
end

function to_xml_schema(::Type{Dict}, properties::Dict)
    tool_parameters = String[]

    ## add parameters
    push!(tool_parameters, "<parameters>")
    for (name, props) in pairs(properties)
        ## sense checks
        name in ["parameter", "parameters"] &&
            @warn "Using a reserved field name: $name. This may cause issues."
        ##
        schema = ["<parameter>",
            "<name>$name</name>"]
        type = get(props, "type", "")
        has_enum = haskey(props, "enum")

        type_lines = if type == "object"
            to_xml_schema(Dict, props["properties"])
        elseif type == "array"
            to_xml_schema(Vector, props["items"])
        elseif has_enum
            to_xml_schema(Enum, props)
        elseif type == ""
            throw(ArgumentError("Type is required for parameter: $name but it was not provided."))
        else
            ["<type>$type</type>"]
        end
        append!(schema, type_lines)
        ## add description
        description = get(props, "description", "")
        !isempty(description) &&
            push!(schema, "<description>$description</description>")
        ## close the parameter
        push!(schema, "</parameter>")
        append!(tool_parameters, schema)
    end
    push!(tool_parameters, "</parameters>")
    return tool_parameters
end

function tool_xml_signature(
        name::String, description::String, properties::Dict{String, Any})
    outputs = String[]
    push!(outputs, "<tool_description>")
    push!(outputs, "<tool_name>$name</tool_name>")
    !isempty(description) &&
        push!(outputs, "<description>$description</description>")
    ## add parameters
    append!(outputs, to_xml_schema(Dict, properties))
    push!(outputs, "</tool_description>")
    return join(outputs, "\n")
end

function tool_xml_signature(datastructtype::Type; max_description_length::Int = 100)
    json_sig = PT.function_call_signature(datastructtype; max_description_length)
    tool_xml_signature(
        json_sig["name"], get(json_sig, "description", ""), json_sig["parameters"]["properties"])
end

function split_tag_value(str::AbstractString)
    splits = split(str, ">")
    if length(splits) == 1
        splits[1], ""
    elseif length(splits) == 2
        splits[1], strip(splits[2])
    else
        throw(ArgumentError("Invalid tag split: $str"))
    end
end

struct XMLParser{T}
    lines::Vector{T}
end
XMLParser(str::AbstractString) = XMLParser(split(str, "<"))
function Base.iterate(itr::XMLParser, state::Integer)
    state > length(itr.lines) && return nothing
    return itr.lines[state], state + 1
end
Base.iterate(itr::XMLParser) = iterate(itr, 1)
Base.eltype(::Type{XMLParser{T}}) where {T} = T
Base.length(itr::XMLParser) = length(itr.lines)

## itr = XMLParser(s)
## for i in itr
##     println(i)
## end

function parse_xml(xml::AbstractString)
    parse_xml(XMLParser(xml))[1]
end

function parse_xml(itr::XMLParser, state::Int = 1, close_tag::AbstractString = "")
    extracted = []
    it = iterate(itr, state)
    while !isnothing(it)
        chunk, state = it
        tag, value = split_tag_value(chunk)
        if isempty(strip(tag))
            ## no tag, skip
            nothing
        elseif tag == close_tag
            ## found closing tag
            ## break and jump up a level
            break
        elseif startswith(tag, "/")
            ## closing tag, ignore - we assume it always matches the opening tag and don't check
            nothing
        elseif !startswith(tag, "/") && isempty(value)
            ## start a new level, eg, "parameters"
            ## go deeper until you find a matching tag
            nested_tags, state = parse_xml(itr, state, "/$tag")
            push!(extracted, tag => nested_tags)
        elseif !isempty(value)
            ## record directly, we assume what follows is the closing tag
            push!(extracted, tag => value)
        else
            @warn "Unexpected situation. Skipping tag: $tag"
        end
        ## next iteration
        it = iterate(itr, state)
    end
    ## transform into output (we need to roll-up repeated tags)
    output = Dict{String, Any}()
    for (tag, value) in extracted
        if haskey(output, tag) && output[tag] isa AbstractVector
            ## if we have multiple tags with the same name, add to a vector
            push!(output[tag], value)
        elseif haskey(output, tag)
            ## change to a vector
            output[tag] = [output[tag], value]
        else
            output[tag] = value
        end
    end

    return output, state
end

"Removes nested `parameters` key which are an artifact from Anthropic XML format."
function flatten_xml_dict(input::AbstractDict)
    ## unnest the parameters key
    if haskey(input, "parameters") && length(keys(input)) == 1
        return flatten_xml_dict(input["parameters"])
    else
        Dict(k => flatten_xml_dict(v) for (k, v) in pairs(input))
    end
end
# Passthrough
flatten_xml_dict(input::Any) = input

"Re-type the input dictionary based on the JSON `schema` provided."
function retype_xml_dict(input::AbstractDict, schema::AbstractDict)
    ## shortcurcuit for singleton types -> they have only 1 key in the schema
    keys_ = collect(keys(schema))
    if keys_ == ["type"]
        return retype_json(schema["type"], input)
    elseif keys_ == ["type", "enum"]
        return retype_json(schema["type"], input)
    end

    out = Dict{String, Any}()
    for (k, v) in pairs(schema)
        if haskey(input, k) && schema[k]["type"] == "object"
            out[k] = retype_xml_dict(input[k], schema[k]["properties"])
        elseif haskey(input, k) && schema[k]["type"] == "array" &&
               input[k] isa AbstractVector
            @info input[k] schema[k]["items"]
            out[k] = retype_xml_dict(input[k], schema[k]["items"])
        elseif haskey(input, k) && schema[k]["type"] == "array"
            ## this is badly formatted array, it is not a vector
            ## @warn "Badly formatted array detected for key $k. Attempting to fix."
            out[k] = retype_xml_dict([input[k]], schema[k]["items"])
        elseif length(v) == 1 && v isa AbstractDict && haskey(v, "type")
            out[k] = retype_json(v["type"], input[k])
        elseif length(v) == 2 && v isa AbstractDict && haskey(v, "type") &&
               haskey(v, "enum")
            out[k] = retype_json(v["type"], input[k])
        elseif k ∉ ["type", "required"]
            throw(ArgumentError("Unknown key $k with value $v"))
            ## out[k] = nothing
        end
    end
    return out
end
# specialization for arrays/vectors
function retype_xml_dict(input::AbstractVector, schema::AbstractDict)
    if collect(keys(schema)) == ["type"] || collect(keys(schema)) == ["type", "enum"]
        [retype_json(schema["type"], val) for val in input]
    elseif haskey(schema, "type") && schema["type"] == "object"
        [retype_xml_dict(val, schema["properties"]) for val in input]
    else
        throw(ArgumentError("Unknown schema $schema versus value $input"))
    end
end
# specialization for singletons
function retype_xml_dict(input::Any, schema::AbstractDict)
    if collect(keys(schema)) == ["type"]
        return retype_json(schema["type"], input)
    else
        throw(ArgumentError("Unknown schema $schema versus value $input"))
    end
end

"Re-type the `value` based on the `type` provided. Reverse operation for singleton values to `to_json_type`"
function retype_json(type::AbstractString, value::AbstractString)
    if type == "string"
        string(value)
    elseif type == "number"
        tryparse(Float64, value)
    elseif type == "integer"
        tryparse(Int, value)
    elseif type == "boolean"
        tryparse(Bool, value)
    elseif type == "null"
        nothing
    else
        throw(ArgumentError("Cannot convert value $value to type $type"))
    end
end
function retype_json(type::AbstractString, value::Number)
    if type == "string"
        string(value)
    elseif type == "number"
        convert(Float64, value)
    elseif type == "integer"
        convert(Int, value)
    elseif type == "boolean"
        convert(Bool, value)
    elseif type == "null"
        nothing
    else
        throw(ArgumentError("Cannot convert value $value to type $type"))
    end
end

function retype_json(type::AbstractString, value::Union{AbstractDict, AbstractVector})
    ## fallback for back XML parser
    if isempty(value) && type == "string"
        ""
    elseif isempty(value) && type == "null"
        nothing
    else
        throw(ArgumentError("Cannot convert value $value to type $type"))
    end
end

"Converts the XML response to the return type. Several try-catch blocks are used to handle errors at different stages."
function xml_to_return_type(content::AbstractString, return_type::Type)
    output = try
        ## select the part with function calls
        xml = split(content, "<function_calls>")[end]
        ## parse the XML into Dict
        tools_dict = parse_xml(xml)
        ## Check if we have multiple invoke statements
        single_tool = if tools_dict["invoke"] isa AbstractVector
            @warn "Multiple tool invocations detected. Only the first one will be processed."
            tools_dict["invoke"][1]["parameters"]
        else
            tools_dict["invoke"]["parameters"]
        end
    catch e
        @warn "Failed to convert the XML response to the return type. Returning the raw response. Error: $e"
        return content
    end

    ## remove nested parameters references
    output = try
        flatten_xml_dict(output)
    catch e
        @warn "Failed to convert the XML response to the return type. Returning the raw response (content: $content) Error: $e"
        return output
    end

    ## Change the types to the JSON schema
    output = try
        json_schema = to_json_schema(return_type)
        retype_xml_dict(output, json_schema["properties"])
    catch e
        @warn "Failed to convert the XML response to the return type. Returning the raw response (content: $content) Error: $e"
        return output
    end

    ## Convert to the return type via JSON3 struct types
    output = try
        output |> JSON3.write |> x -> JSON3.read(x, return_type)
    catch e
        @warn "Failed to convert the XML response to the return type. Returning the raw response (content: $content) Error: $e"
        return output
    end

    return output
end
######################
# Useful Structs
######################

# This is kindly borrowed from the awesome Instructor package](https://github.com/jxnl/instructor/blob/main/instructor/dsl/maybe.py).
"""
Extract a result from the provided data, if any, otherwise set the error and message fields.

# Arguments
- `error::Bool`: `true` if a result is found, `false` otherwise.
- `message::String`: Only present if no result is found, should be short and concise.
"""
struct MaybeExtract{T <: Any}
    result::Union{Nothing, T}
    error::Bool
    message::Union{Nothing, String}
end

"""
Extract zero, one or more specified items from the provided data.
"""
struct ItemsExtract{T <: Any}
    items::Vector{T}
end
