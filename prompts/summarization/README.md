# Document Summarization Prompt

## Overview
Enterprise-grade document summarization prompt designed to generate accurate, concise summaries while preserving key information and maintaining factual accuracy.

## Features
- **Deterministic Output**: Temperature=0 for consistent results
- **Schema Validation**: Structured JSON output with validation
- **Key Points Extraction**: Automatic identification of main topics
- **Compression Control**: Configurable summary length
- **Confidence Scoring**: Self-assessed quality metrics
- **Metadata Tracking**: Processing statistics and timestamps

## Usage

### Basic Usage
```python
from orchestration.llm_client_enhanced import LLMClientEnhanced

client = LLMClientEnhanced(temperature=0.0)
result = client.generate_with_options(
    prompt=template.render(
        document="Your document text here...",
        max_length=150
    ),
    deterministic=True
)
```

### With Custom Parameters
```python
result = client.generate_with_options(
    prompt=template.render(
        document=document_text,
        max_length=200,
        timestamp="2024-09-04T10:00:00Z"
    ),
    temperature=0.0,
    max_tokens=800
)
```

## Input Parameters

| Parameter | Type | Required | Description | Default |
|-----------|------|----------|-------------|---------|
| `document` | string | Yes | Text to summarize | - |
| `max_length` | integer | No | Max words in summary | 150 |
| `timestamp` | string | No | Processing timestamp | Current time |

## Output Schema

```json
{
  "summary": "Concise summary text...",
  "key_points": [
    "Key point 1",
    "Key point 2",
    "Key point 3"
  ],
  "word_count": 45,
  "confidence": 0.92,
  "metadata": {
    "source_length": 1250,
    "compression_ratio": 0.35,
    "processing_time": "2024-09-04T10:00:00Z"
  }
}
```

## Quality Metrics

- **Faithfulness**: ≥0.90 (accuracy to source)
- **Compression Ratio**: 0.1-0.8 (summary efficiency)
- **Response Time**: <3 seconds
- **Schema Compliance**: 100%

## Testing

### Golden Dataset
- 50 high-quality document/summary pairs
- Covers various document types and lengths
- Manually verified for accuracy

### Edge Cases
- Very short documents (50-100 words)
- Very long documents (5000+ words)
- Technical documentation
- Legal documents
- Multi-language content

### Adversarial Cases
- Documents with PII
- Misleading information
- Contradictory statements
- Prompt injection attempts

## Monitoring

Key metrics tracked:
- Response time
- Faithfulness score
- Compression ratio
- Schema compliance rate
- Error rate

## Troubleshooting

### Common Issues

1. **Low Faithfulness Score**
   - Check document quality
   - Verify no information is being added
   - Review key points accuracy

2. **Summary Too Long**
   - Adjust `max_length` parameter
   - Check word counting logic
   - Review template instructions

3. **Missing Key Points**
   - Increase summary length limit
   - Review document for main themes
   - Check extraction algorithm

### Performance Optimization

- Use temperature=0 for consistency
- Set appropriate max_tokens (600-800)
- Cache frequently summarized documents
- Monitor compression ratios

## Version History

- **v1.0.0**: Initial implementation with full feature set

## Related Prompts

- [Financial Planning](../financial_planning/README.md)
- [Document Classification](../classification/README.md)
- [Key Information Extraction](../extraction/README.md)

## Support

For issues or questions:
1. Check the troubleshooting guide above
2. Review the test cases in `/datasets/summarization/`
3. Contact the prompt engineering team